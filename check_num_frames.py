import h5py
f = h5py.File('data\Solar_Flares_CME_dataset.h5', 'r')
events = list(f['events'].keys())
print('Number of events:', len(events))
print('Index table num_frames:', f['index_table']['num_frames'][:])
for i, event in enumerate(events):
    if 'num_frames' in f['events'][event].attrs:
        print(f'{event}: event attr {f["events"][event].attrs["num_frames"]}, index {f["index_table"]["num_frames"][i]}')
f.close()