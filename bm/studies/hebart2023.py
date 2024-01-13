import typing as tp
from itertools import product
from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from . import api
from . import utils
from .download import download_osf
from ..events import extract_sequence_info

class StudyPaths(utils.StudyPaths):
    def __init__(self) -> None:
        super().__init__(Hebart2023Recording.study_name())
        self.megs = self.download / "all_data" / "MEG"
        # self.events = self.download / "stimuli" / "events"


class Hebart2023Recording(api.Recording):
    data_url = "https://new-data-url.com"  # Update with the correct URL
    paper_url = "https://new-paper-url.com"  # Update with the correct URL
    doi = "https://new-doi-url.com"  # Update with the correct URL
    licence = ''
    modality = "visual"  # Update if necessary
    language = "en"  # Update if necessary
    device = "meg"  # Update if necessary
    description = "New study description."  # Update with the correct description

    @classmethod
    def download(cls) -> None:
        # Update with the correct download instructions
        pass

    @classmethod
    def iter(cls) -> tp.Iterator["Hebart2023Recording"]:  # type: ignore
        """Returns a generator of all recordings"""
        # Update with the correct iteration instructions

        # download, extract, organize
        cls.download()
        # List all recordings: depends on study structure
        paths = StudyPaths()
        # print(paths)
        # print(paths.download)
        # import pdb; pdb.set_trace()
        subject_file = paths.download / "participants.tsv"
        subjects = pd.read_csv(subject_file, sep="\t")
        def get_subject_id(x):
            return x.split("-")[1]  # noqa

        subjects = subjects.participant_id.apply(get_subject_id).values
        # stories = [str(x) for x in range(1)]
        tasks = ['main']
        sessions = ['{:02d}'.format(x) for x in range(1,13)]  # 2 recording sessions
        runs = ['{:02d}'.format(x) for x in range(1,11)]
        for subject, session, task, run in product(subjects, sessions, tasks, runs):
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                root=paths.download,
                run = run,
                datatype="meg",
            )
            if not Path(str(bids_path)).exists():
                continue
            # import pdb; pdb.set_trace()
            recording = cls(subject_uid=subject, session=session, task=task, run=run)
            yield recording

    def __init__(self, subject_uid: str, session: str, task: str, run: str) -> None:
        recording_uid = f'{subject_uid}_session{session}_task{task}'
        super().__init__(subject_uid=subject_uid, recording_uid=recording_uid)
        # Update with the correct initialization instructions
        self.task = task
        self.session = session
        self.run = run

    def _load_raw(self) -> mne.io.RawArray:
        # Update with the correct raw data loading instructions
        paths = StudyPaths()
        bids_path = BIDSPath(
            subject=self.subject_uid,
            session=self.session,
            task=self.task,
            root=paths.download,
            run = self.run,
            datatype="meg",
        )
        raw = read_raw_bids(bids_path)  # FIXME this is NOT a lazy read
        self.raw_sample_rate = raw.info["sfreq"]
        picks = dict(meg=True, eeg=False, stim=False, eog=False, ecg=False, misc=False)
        raw = raw.pick_types(**picks)
        return raw

    def _load_events(self) -> pd.DataFrame:
        """
        in this particular data, I'm transforming our original rich dataframe
        into mne use a Annotation class in order to save the whole thing into
        a *.fif file, At reading time, I'm converting it back to a DataFrame
        """
        # Update with the correct event data loading instructions
        
        raw = self.raw()
        paths = StudyPaths()
        # extract annotations
        events = list()
        for annot in raw.annotations:
            # event = eval(annot.pop("description"))
            event = dict()
            event['kind'] = annot['description'].split('/')[0]
            if annot['description'].split('/')[0] != 'catch':
                event['image_id'] = annot['description'].split('/')[1]
            else:
                event['image_id'] = '-1' # catch trials, maybe we should change this
            event['start'] = annot['onset']
            event['duration'] = annot['duration']
            events.append(event)
            
        # import pdb; pdb.set_trace()
        events_df = pd.DataFrame(events)
            #  kind  image_id       start  duration
            # 0     exp   24812.0    2.854167       0.5
            # 1     exp    8355.0    4.505000       0.5
            # 2     exp    9054.0    6.055833       0.5
            # 3     exp    3648.0    7.405833       0.5
            # 4     exp   21998.0    8.923333       0.5
            # ..    ...       ...         ...       ...
            # 221   exp    2869.0  334.268333       0.5
            # 222   exp   14257.0  335.751667       0.5
            # 223   exp   19632.0  337.319167       0.5
            # 224   exp   20512.0  338.953333       0.5
            # 225  test   25073.0  340.604167       0.5
        events_df[['language', 'modality']] = 'english', 'visual'
        events_df = events_df.event.create_blocks(groupby='exp')

        return events_df