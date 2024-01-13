# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections import namedtuple
from concurrent import futures
import logging
import typing as tp
from pathlib import Path

from dora.log import LogProgress
import flashy
import pandas as pd
import numpy as np
import mne
import torch
from torch.utils.data import ConcatDataset
from torch.nn import functional as F

from . import env
from .cache import Cache
from . import studies
from .features import FeaturesBuilder
from .events import Event, split_wav_as_block, assign_blocks
from .utils import Frequency, roundrobin

import pdb

# pylint: disable=logging-fstring-interpolation
logger = logging.getLogger(__name__)
OptionalPath = tp.Optional[tp.Union[str, Path]]
# pylint: disable=too-few-public-methods,too-many-arguments,no-self-use,too-many-instance-attributes


class _DatasetFactory:
    """Defines how to extract epochs.

    Parameters
    ----------
    condition: str or float
        Either an event in the condition columns of metadata, or a stride in seconds, or a full
        pandas query (if it contains "=").
    tmin: float
        Start time with respect to the events.
    tmax: float
        End time with respect to the events.
    decim: int
        Factor by which to subsample the data.
    baseline: tuple
        Baseline parameter as in mne.Epochs.

    Note
    ----
    See full descriptions tmin, tmax, decim and baseline parameters in the MNE documentation:
    https://mne.tools/stable/generated/mne.Epochs.html#mne-epochs
    """

    # pylint: disable=unused-argument,function-redefined
    def __init__(
            self,
            condition: tp.Union[str, float] = 3.0,
            tmin: float = -0.5,
            tmax: float = 2.5,
            baseline: tp.Any = (None, 0),
            decim: int = 1,
            sample_rate: float = studies.schoffelen2019.RAW_SAMPLE_RATE,
            highpass: float = 0,
            features: tp.Sequence[str] = ("WordLength", "WordFrequency"),
            features_params: tp.Optional[dict] = None,
            ignore_end_in_block: bool = False,
            ignore_start_in_block: bool = False,
            event_mask: bool = False,
            split_wav_as_block: bool = False,
            meg_dimension: tp.Optional[int] = None,
            autoreject: bool = False,
    ) -> None:
        assert tmin < tmax
        assert decim == 1, "Decimation factor is not yet supported"
        self.features = list(features)
        self.features_params = features_params
        self.condition = condition
        self.baseline = baseline
        self.sample_rate = int(round(sample_rate))
        self.highpass = highpass
        self.ignore_end_in_block = ignore_end_in_block
        self.ignore_start_in_block = ignore_start_in_block
        self.event_mask = event_mask
        self.meg_dimension = meg_dimension
        self.split_wav_as_block = split_wav_as_block
        self.autoreject = autoreject
        self._opts = dict(tmin=tmin, tmax=tmax, decim=decim)

    # pylint: disable=too-many-locals
    def apply(
        self, recording: studies.Recording,
        blocks: tp.Optional[tp.List[tp.Tuple[float, float]]] = None
    ) -> tp.Optional["SegmentDataset"]:
        """Apply the epochs extraction procedure to the raw file and create a SegmentDataset.

        Parameters
        ----------
        recording:
            Recording on which to apply the epochs definition.
        blocks:
            List of tuples representing available ranges for building the dataset.
        """
        if blocks is not None and not blocks:
            raise ValueError("No blocks provided.")
        data = recording.preprocessed(self.sample_rate, highpass=self.highpass)
        sample_rate = Frequency(data.info["sfreq"])
        assert int(sample_rate) == int(self.sample_rate)
        if isinstance(self.condition, str):
            # hack to discriminate between a condition and a query
            query = self.condition if "=" in self.condition else f"kind=={self.condition!r}"
            meta = recording.events().query(query)
            times = meta.start.values
        elif isinstance(self.condition, float):
            # Define events every x seconds
            times = np.arange(0, data.times[-1], self.condition)
            meta = None
        else:
            raise TypeError("Condition should be a string, or a float corresponding to a "
                            "duration in seconds, got:\n"
                            f"{self.condition} (type: {type(self.condition)})")

        events = recording.events().copy()
        events = events.sort_values('start')
        if self.split_wav_as_block:
            assert blocks is not None
            events = split_wav_as_block(events, blocks)

        delta = 0.5 / sample_rate
        mask = np.logical_and(times + self._opts["tmin"] >= 0,
                              times + self._opts["tmax"] < data.times[-1] + delta)
        if blocks is not None:
            # We only keep extracts that are fully contained in at least one of the given blocks.
            in_any_split = False
            for start, stop in blocks:
                if self.ignore_start_in_block:
                    in_split = times >= start
                else:
                    in_split = times + self._opts["tmin"] >= start
                margin = delta if self.ignore_end_in_block else self._opts["tmax"] - delta
                in_split &= times + margin < stop
                # Keep around for debugging
                # print("block", start, stop)
                # print("need", start - self._opts["tmin"], stop + delta - self._opts["tmax"])
                # print("ok", sum(in_split))
                in_any_split |= in_split
                # import pdb; pdb.set_trace()
            mask &= in_any_split
        if not mask.any():
            logger.warning("Empty dataset %r", recording)
            return None
        # assert mask.any(), "empty dataset"
        # TODO understand why samples/times some are not unique nor ordered
        # import pdb; pdb.set_trace()
        samples = sample_rate.to_ind(times[mask])
        unique_samples = np.unique(samples)
        if len(unique_samples) != len(samples):
            logger.warning(f"Found {len(samples) - len(unique_samples)} duplicates out of "
                           f"{len(samples)} events")
        if len(np.where(np.diff(times[mask]) < 0)[0]) > 0:
            logger.warning(f"Times are not sorted in meg events data at indices "
                           f"{np.where(np.diff(times[mask]) < 0)[0]}. "
                           f"SubjectID={recording.subject_uid}")

        if meta is not None:
            meta = meta.iloc[np.where(mask)].reset_index()
        mne_events = np.concatenate([samples[:, None], np.ones(
            (len(samples), 2), dtype=np.int64)], 1)  # why long?
        # create
        baseline = self.baseline
        epochs = mne.Epochs(data, events=mne_events,
                            preload=False, baseline=baseline,
                            metadata=meta, **self._opts, event_repeated='drop')
        epochs._bad_dropped = True  # Hack: avoid checking
        if self.autoreject:
            from .autoreject import AutoRejectDrop

            raw = epochs._raw
            autoreject_cache = Cache('autoreject', args=(self.__dict__, blocks))

            def _get_autoreject():
                logger.info('Computing autoreject, cachefile %s', autoreject_cache.cache_path({}))
                num_samples = 200
                gen = torch.Generator()
                gen.manual_seed(1234)
                indexes = torch.randperm(len(epochs))[:num_samples].tolist()
                epochs.load_data()
                autoreject = AutoRejectDrop()
                autoreject.fit(epochs[indexes])
                return autoreject

            autoreject = autoreject_cache.get(_get_autoreject)
            epochs.load_data()
            new_epochs = autoreject.transform(epochs)
            assert len(new_epochs) == len(epochs), (len(new_epochs), len(epochs))
            epochs = new_epochs
            epochs._raw = raw

        dset = SegmentDataset(
            recording, epochs, events=events,
            features=self.features, features_params=self.features_params,
            event_mask=self.event_mask, meg_dimension=self.meg_dimension)
        dset.blocks = blocks  # type: ignore
        return dset


@dataclasses.dataclass
class SegmentBatch:
    """Collatable training data."""
    meg: torch.Tensor
    features: torch.Tensor
    features_mask: torch.Tensor
    subject_index: torch.Tensor
    recording_index: torch.Tensor
    # optional for now
    _recordings: tp.List[studies.Recording] = dataclasses.field(default_factory=list)
    _event_lists: tp.List[tp.List[Event]] = dataclasses.field(default_factory=list)

    def to(self, device: tp.Any) -> "SegmentBatch":
        """Creates a new instance on the appropriate device."""
        out: tp.Dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if isinstance(data, torch.Tensor):
                out[field.name] = data.to(device)
            else:
                out[field.name] = data
        return SegmentBatch(**out)

    def replace(self, **kwargs) -> "SegmentBatch":
        cls = self.__class__
        kw = {}
        for field in dataclasses.fields(cls):
            if field.name in kwargs:
                kw[field.name] = kwargs[field.name]
            else:
                kw[field.name] = getattr(self, field.name)
        return cls(**kw)

    def __getitem__(self, index) -> "SegmentBatch":
        cls = self.__class__
        kw = {}
        indexes = torch.arange(
            len(self), device=self.meg.device)[index].tolist()  # explicit indexes for lists
        for field in dataclasses.fields(cls):
            data = getattr(self, field.name)
            if isinstance(data, list):
                if data:
                    value = [data[idx] for idx in indexes]
                else:
                    value = []
            else:
                value = data[index]
            kw[field.name] = value
        return cls(**kw)

    def __len__(self) -> int:
        return len(self.meg)

    @classmethod
    def collate_fn(cls, meg_features_list: tp.List["SegmentBatch"]) -> "SegmentBatch":
        out: tp.Dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(cls):
            data = [getattr(mf, field.name) for mf in meg_features_list]
            if isinstance(data[0], torch.Tensor):
                out[field.name] = torch.stack(data)
            else:
                out[field.name] = [x for y in data for x in y]
        meg_features = SegmentBatch(**out)
        # check that list sizes are either 0 or batch size
        batch_size = meg_features.meg.shape[0]
        for field in dataclasses.fields(meg_features):
            val = out[field.name]
            if isinstance(val, list):
                assert len(val) in (0, batch_size), f"Incorrect size for {field.name}"
        return meg_features


class SegmentDataset:
    """Iterable over epochs of MEG data and features.

    The annotations are embedded lazily from the metadata for each epoch.

    Note
    ----
    These should be instantiated through a factory with epochs extraction information, in order to
    avoid moving around so many parameters.

    Example
    -------

        recording = next(bm.studies.register["Schoffelen2019"].iter())
        factory = bm.SegmentDataset.Factory(condition="word", tmin=-0.5, tmax=0.5)
        dataset = factory.apply(recording)
        meg_var, features_var, index = next(iter(dataset))

    """

    Factory = _DatasetFactory

    def __init__(self, recording: studies.Recording, epochs: mne.Epochs,
                 features: tp.Sequence[str], events: pd.DataFrame,
                 features_params: tp.Optional[dict] = None, event_mask: bool = False,
                 meg_dimension: tp.Optional[int] = None) -> None:
        self.recording = recording
        self.epochs = epochs
        self.events = events
        self.sample_rate = Frequency(epochs._raw.info["sfreq"])
        self.features_params = features_params
        features_params_dict = dict(
            self.features_params) if features_params else {}  # type: ignore
        self.features = FeaturesBuilder(
            events, features,
            features_params=features_params_dict,
            sample_rate=self.sample_rate,
            event_mask=event_mask)
        self.meg_dimension = meg_dimension
        if meg_dimension is not None:
            assert meg_dimension >= self.recording.meg_dimension

    def _get_bounds_times(self, idx: int) -> tp.Tuple[float, float]:
        """Infers the start and stop times of a given epoch
        """
        ep = self.epochs
        # from mne code
        event_samp = ep.events[idx, 0]
        sample_rate = self.sample_rate
        start = event_samp + sample_rate.to_ind(ep._raw_times[0])
        start -= ep._raw.first_samp  # offset
        stop = start + len(ep._raw_times)
        return (sample_rate.to_sec(start), sample_rate.to_sec(stop))

    def _get_full_feature(self) -> torch.Tensor:
        """Creates the full array of features (useful for testing)
        """
        return self.features(0, self.sample_rate.to_sec(self.epochs._raw.n_times))[0]

    def _get_feature(self, idx: int) -> torch.Tensor:
        """Get the feature corresponding to index idx
        """
        start, stop = self._get_bounds_times(idx)
        return self.features(start, stop)

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, index: tp.Any) -> tp.Any:
        if isinstance(index, int):
            meg = next(self.epochs[index])
            meg_torch = torch.from_numpy(meg).float()
            if self.meg_dimension is not None:
                meg_torch = F.pad(meg_torch, (0, 0, 0, self.meg_dimension - meg_torch.shape[0]))
            feature_data, feature_mask, events = self._get_feature(index)
            return SegmentBatch(
                meg=meg_torch,
                features=feature_data,
                features_mask=feature_mask,
                subject_index=torch.tensor(self.recording.subject_index),
                recording_index=torch.tensor(self.recording.recording_index),
                _recordings=[self.recording.empty_copy()],  # don't copy meta and meg
                _event_lists=[events],
            )
        else:
            features = list(self.features.keys())
            return self.__class__(
                self.recording, self.epochs[index], events=self.events,
                features=features, features_params=self.features_params)

    def __iter__(self) -> tp.Iterator[SegmentBatch]:
        return (self[k] for k in range(len(self)))  # pleases mypy


Datasets = namedtuple("Datasets", "train valid test")


def _preload(recording: studies.Recording, **kwargs: tp.Any) -> studies.Recording:
    """Calls cached data to create it if need be
    """
    recording.events()
    recording.preprocessed(**kwargs)
    return recording


def _extract_recordings(selections: tp.List[tp.Dict[str, tp.Any]], n_recordings: int,
                        skip_recordings: int = 0, shuffle_recordings_seed: int = -1
                        ) -> tp.Sequence[studies.Recording]:
    """Extract the number of recordings required, and mix audio and visual if need be
    """  # this is a function to help testing, especially the "any" case
    recording_lists = [list(studies.from_selection(select)) for select in selections]
    # print(recording_lists)
    # import pdb; pdb.set_trace()
    if shuffle_recordings_seed > 0:  # deactivated if -1
        rng = np.random.RandomState(seed=shuffle_recordings_seed)
        for subjs in recording_lists:
            rng.shuffle(subjs)  # type: ignore
    all_recordings = list(roundrobin(*recording_lists))
    all_recordings = all_recordings[skip_recordings: skip_recordings + n_recordings]
    # print(all_recordings)
    # import pdb; pdb.set_trace()
    if len(all_recordings) < n_recordings:
        logger.warning("Requested %d recordings but only found %d",
                       n_recordings, len(all_recordings))
    # assign subject index
    uids = sorted(set((r.__class__.__name__, r.subject_uid) for r in all_recordings))
    uids_index = {uid: k for k, uid in enumerate(uids)}
    for r_index, r in enumerate(all_recordings):
        index = uids_index[(r.__class__.__name__, r.subject_uid)]
        assert r._subject_index in (None, index), "Cannot assign a different index"
        r._subject_index = index
        r._recording_index = r_index
    return all_recordings


def get_datasets(
        selections: tp.List[tp.Dict[str, tp.Any]],
        n_recordings: int,
        test_ratio: float,
        valid_ratio: float,
        sample_rate: int = studies.schoffelen2019.RAW_SAMPLE_RATE,  # FIXME
        highpass: float = 0,
        # num_workers: int = 10,
        num_workers: int = 1,
        apply_baseline: bool = True,
        progress: bool = False,
        skip_recordings: int = 0,
        min_block_duration: float = 0.0,
        force_uid_assignement: bool = True,
        shuffle_recordings_seed: int = -1,
        split_assign_seed: int = 12,
        min_n_blocks_per_split: int = 20,
        features: tp.Optional[tp.List[str]] = None,
        extra_test_features: tp.Optional[tp.List[str]] = None,
        test: dict = {},
        allow_empty_split: bool = False,
        n_subjects: tp.Optional[int] = None,
        n_subjects_test: tp.Optional[int] = None,
        remove_ratio: float = 0.,
        **factory_kwargs: tp.Any) -> Datasets:
    """
    """
    if features is None:
        features = []
    if extra_test_features is None:
        extra_test_features = []
    assert env.cache is not None
    num_workers = max(1, min(n_recordings, num_workers))
    print(f'num_workers: {num_workers}')

    # Use barrier to prevent multiple workers from computing the cache
    # in parallel.
    if not flashy.distrib.is_rank_zero():
        flashy.distrib.barrier()  # type: ignore
    # get recordings
    all_recordings = _extract_recordings(
        selections, n_recordings, skip_recordings=skip_recordings,
        shuffle_recordings_seed=shuffle_recordings_seed)
    # import pdb; pdb.set_trace()
    if num_workers <= 1:
        if progress:
            all_recordings = LogProgress(logger, all_recordings,   # type: ignore
                                         name="Preparing cache", level=logging.DEBUG)
        all_recordings = [  # for debugging
            _preload(s, sample_rate=sample_rate, highpass=highpass) for s in all_recordings]
    else:
        # precompute slow metadata loading
        
        with futures.ProcessPoolExecutor(num_workers) as pool:
            jobs = [pool.submit(_preload, s, sample_rate=sample_rate, highpass=highpass)
                    for s in all_recordings]
            if progress:
                jobs = LogProgress(logger, jobs, name="Preparing cache",  # type: ignore
                                   level=logging.DEBUG)
            all_recordings = [j.result() for j in jobs]  # check for exceptions
    if flashy.distrib.is_rank_zero():
        flashy.distrib.barrier()  # type: ignore
    # import pdb; pdb.set_trace()
    # create datasets through factory, split them and concatenate
    meg_dimension = max(recording.meg_dimension for recording in all_recordings)
    factory_kwargs.update(sample_rate=sample_rate, highpass=highpass, meg_dimension=meg_dimension,
                          baseline=(None, 0) if apply_baseline else None)
    fact = SegmentDataset.Factory(features=features, **factory_kwargs)
    for key, value in test.items():
        if value is not None:
            factory_kwargs[key] = value
    fact_test = SegmentDataset.Factory(features=features + extra_test_features, **factory_kwargs)

    factories = [fact_test, fact, fact]

    n_recordings = len(all_recordings)
    if progress:
        all_recordings = LogProgress(
            logger, all_recordings, name="Loading Subjects")  # type: ignore[assignment]

    dsets_per_split: tp.List[tp.List[SegmentDataset]] = [[], [], []]
    for i, recording in enumerate(all_recordings):
        # import pdb; pdb.set_trace()
        events = recording.events()
        #      kind  image_id      start  duration language modality      uid
        # 0   block       NaN   2.854167       1.0  english   visual  24812.0
        # 1     exp   24812.0   2.854167       0.5  english   visual      NaN
        # 2   block       NaN   4.505000       1.0  english   visual   8355.0
        # 3     exp    8355.0   4.505000       0.5  english   visual      NaN
        # 4   block       NaN   6.055833       1.0  english   visual   9054.0
        # 5     exp    9054.0   6.055833       0.5  english   visual      NaN
        # 6   block       NaN   7.405833       1.0  english   visual   3648.0
        # 7     exp    3648.0   7.405833       0.5  english   visual      NaN
        # 8   block       NaN   8.923333       1.0  english   visual  21998.0
        # 9     exp   21998.0   8.923333       0.5  english   visual      NaN
        # 10  block       NaN  10.407500       1.0  english   visual   4803.0
        # 11    exp    4803.0  10.407500       0.5  english   visual      NaN
        # import pdb; pdb.set_trace()
        blocks = events[events.kind == 'block']

        if min_block_duration > 0 and not force_uid_assignement:
            if recording.study_name() not in ['schoffelen2019', 'hebart2023']:
                blocks = blocks.event.merge_blocks(min_block_duration_s=min_block_duration)

        blocks = assign_blocks(
            blocks, [test_ratio, valid_ratio], remove_ratio=remove_ratio, seed=split_assign_seed,
            min_n_blocks_per_split=min_n_blocks_per_split)
        # (Pdb) p blocks[blocks['split'] == 0].shape
        # (33, 8)
        # (Pdb) p blocks[blocks['split'] == 1].shape
        # (18, 8)
        # (Pdb) p blocks[blocks['split'] == 2].shape
        # (135, 8)
        
        # import pdb; pdb.set_trace()

        for j, (fact, dsets) in enumerate(zip(factories, dsets_per_split)):
            split_blocks = blocks[blocks.split == j]
            if not split_blocks.empty:
                start_stops = [(b.start, b.start + b.duration) for b in split_blocks.itertuples()]
                # [(19.310833, 22.195), (60.860833, 62.394167), (65.445833, 67.146667), (89.421667, 91.005833), (114.715, 117.715833), (141.575833, 142.859167), (146.960833, 148.411667), (148.411667, 150.045833), (154.414167, 155.764167), (161.916667, 163.400833), (163.400833, 164.968333), (166.518333, 167.969167), (184.875833, 186.36), (192.579167, 194.079167), (194.079167, 195.513333), (195.513333, 197.064167), (203.083333, 204.416667), (210.0025, 213.07), (233.728333, 236.546667), (236.546667, 238.13), (238.13, 241.398333), (257.655, 259.221667), (286.2825, 287.733333), (292.535, 293.885833), (297.036667, 298.454167), (301.705, 303.405833), (304.973333, 306.5575), (306.5575, 308.124167), (308.124167, 309.475), (315.2275, 316.5775), (316.5775, 317.978333), (327.181667, 328.6325), (334.268333, 335.751667)]
                # [(18.810833, 19.810833), (60.360833, 61.360833), (64.945833, 65.945833), (88.921667, 89.921667), (114.215, 115.215), (141.075833, 142.075833), (146.460833, 147.460833), (147.911667, 148.911667), (153.914167, 154.914167), (161.416667, 162.416667), (162.900833, 163.900833), (166.018333, 167.018333), (184.375833, 185.375833), (192.079167, 193.079167), (193.579167, 194.579167), (195.013333, 196.013333), (202.583333, 203.583333), (209.5025, 210.5025), (233.228333, 234.228333), (236.046667, 237.046667), (237.63, 238.63), (257.155, 258.155), (285.7825, 286.7825), (292.035, 293.035), (296.536667, 297.536667), (301.205, 302.205), (304.473333, 305.473333), (306.0575, 307.0575), (307.624167, 308.624167), (314.7275, 315.7275), (316.0775, 317.0775), (326.681667, 327.681667), (333.768333, 334.768333)]
                # [(19.310833, 20.310833), (60.860833, 61.860833), (65.445833, 66.445833), (89.421667, 90.421667), (114.715, 115.715), (141.575833, 142.575833), (146.960833, 147.960833), (148.411667, 149.411667), (154.414167, 155.414167), (161.916667, 162.916667), (163.400833, 164.400833), (166.518333, 167.518333), (184.875833, 185.875833), (192.579167, 193.579167), (194.079167, 195.079167), (195.513333, 196.513333), (203.083333, 204.083333), (210.0025, 211.0025), (233.728333, 234.728333), (236.546667, 237.546667), (238.13, 239.13), (257.655, 258.655), (286.2825, 287.2825), (292.535, 293.535), (297.036667, 298.036667), (301.705, 302.705), (304.973333, 305.973333), (306.5575, 307.5575), (308.124167, 309.124167), (315.2275, 316.2275), (316.5775, 317.5775), (327.181667, 328.181667), (334.268333, 335.268333)]
                # import pdb; pdb.set_trace()
                dset = fact.apply(recording, blocks=start_stops)
                if dset is not None:
                    dsets.append(dset)
                else:
                    logger.warning(f'Empty blocks for split {j + 1}/{len(factories)} of '
                                   f'recording {i + 1}/{n_recordings}.')
            else:
                logger.warning(f'No blocks found for split {j + 1}/{len(factories)} of '
                               f'recording {i + 1}/{n_recordings}.')

    # import pdb; pdb.set_trace()

    if not allow_empty_split:
        empty_names = [name for name, dset in zip(
            ['train', 'valid', 'test'], dsets_per_split[::-1]) if len(dset) == 0]
        if empty_names:
            raise ValueError(f'The following splits are empty: {empty_names}.')

    # Select subset of subjects
    testset, validset, trainset = dsets_per_split
    if n_subjects:
        seen_subjects = set()
        count = 0
        for dset in trainset:
            seen_subjects.add(dset.recording.subject_uid)
            if len(seen_subjects) > n_subjects:
                break
            count += 1
        validset = validset[:count]
        trainset = trainset[:count]
    if n_subjects_test:
        seen_subjects = set()
        count = 0
        for dset in testset:
            seen_subjects.add(dset.recording.subject_uid)
            if len(seen_subjects) > n_subjects_test:
                break
            count += 1
        testset = testset[:count]

    splits = [ConcatDataset(dset) for dset in dsets_per_split[::-1]]
    msg = '# Examples (train | valid | test): ' + ' | '.join([str(len(dset)) for dset in splits])
    logger.info(msg)

    return Datasets(*splits)
