from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('DAMO_NLP/jd', subset_name='default', split='train')

print(ds)