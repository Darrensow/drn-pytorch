import torch

def train(model, train_loader, criterion, optimizer, epochs, epoch):
  model.train()
  running_loss = 0.0
  for batch_idx, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    # batch_idx += 1
    # if batch_idx % 10 == 0:
    #   print(f'Epoch [{epoch+1}/{epochs}]  Batch [{batch_idx}/{len(train_loader)}]  Loss: {running_loss / batch_idx:.6f}')

  epoch_loss = running_loss / len(train_loader)
  print(f'Epoch [{epoch+1}/{epochs}]  Training Loss: {epoch_loss:.6f}')
  return epoch_loss

def validate(model, val_loader, criterion, epochs, epoch):
  model.eval()
  running_loss = 0.0
  num_samples = 0
  for batch_idx, (inputs, targets) in enumerate(val_loader):
    with torch.no_grad():
      outputs = model(inputs)
    loss = criterion(outputs, targets)
    running_loss += loss.item()
    num_samples += targets.size(0)
    # batch_idx += 1
    # if batch_idx % 10 == 0:
    #   print(f'Epoch [{epoch+1}/{epochs}]  Batch [{batch_idx}/{len(val_loader)}]  Loss: {running_loss / batch_idx:.6f}')

  epoch_loss = running_loss / len(val_loader)
  print(f'Epoch [{epoch+1}/{epochs}] Validation Loss: {running_loss/len(val_loader):.6f}')
  return epoch_loss