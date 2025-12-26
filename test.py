from data_loader import EEGGraphDataset
import mne
raw = mne.io.read_raw_eeglab(
        r"C:\Users\kira7\Downloads\ds004504-main\ds004504-main\sub-048\eeg\sub-048_task-eyesclosed_eeg.set",
        preload=True,

)   
