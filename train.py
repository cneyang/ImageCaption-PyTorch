from utils.utils import *

torch.manual_seed(17)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Dataset
dataset = ImageCaptionDataset(path="data", train=True)
train_set, val_set = random_split(dataset, [round(0.8*len(dataset)), round(0.2*len(dataset))])

# DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

# Pretrained Encoder Model
encoder = Encoder()
decoder = Decoder(dataset.vocab.num_words)

model = ImageCaptioningModel(encoder, decoder, device).to(device)

# optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.NLLLoss(ignore_index=2).to(device)

# Train pretrained model
min_loss = 1000

epochs = 30
for epoch in range(epochs):
    train_loss = 0.0
    for x, y in tqdm(train_loader):
        output = model(x, y)

        output_dim = output.size(-1)
        output = output.view(-1, output_dim)
        y = y.view(-1).to(device)

        optimizer.zero_grad()
        loss = criterion(output, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        train_loss += loss

    print("\rEpoch: {} Train Loss: {:.2f}".format(epoch+1, train_loss/len(train_loader)))

    # Validation
    with torch.no_grad():
        val_loss = 0.0
        for x, y in tqdm(val_loader):
            output = model(x, y)

            output_dim = output.size(-1)
            output = output.view(-1, output_dim)
            y = y.view(-1).to(device)

            loss = criterion(output, y)

            val_loss += loss

        print("\rVal Loss: {:.2f}".format(val_loss / len(val_loader)))

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), "./model.pt")
