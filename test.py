from utils.utils import *

torch.manual_seed(17)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Dataset
dataset = ImageCaptionDataset(train=False)
test_set = ImageCaptionDataset(path="test", train=False)

test_loader = DataLoader(test_set, batch_size=1)

encoder = Encoder()
decoder = Decoder(dataset.vocab.num_words)

model = ImageCaptioningModel(encoder, decoder, device).to(device)

model.load_state_dict(torch.load("./model.pt"))

inference_dict = dict()
inference_dict['images'] = list()

model.eval()
with torch.no_grad():
    for x, img_name in tqdm(test_loader):
        caption_dict = dict()
        captions = model.predict(x, dataset)

        captions = " ".join([word for word in captions])
        caption_dict['file_name'] = img_name[0]
        caption_dict['captions'] = captions

        inference_dict['images'].append(caption_dict)

with open("result.json", "w") as json_file:
    json.dump(inference_dict, json_file)
