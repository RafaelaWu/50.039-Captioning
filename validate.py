import torch
import argparse
import pickle 
from data_loader import get_loader 
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import sentence_bleu
import torch.nn as nn
import time
import pdb
import math
from pathlib import Path

home = str(Path.home())

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def val(args):
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    print('val_loader length = {}'.format(len(data_loader)))
    
    reference = list()
    sb1,sb2,sb3,sb4=0,0,0,0
    val_accuracy= 0
    val_loss = 0
    start = time.time()
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # Forward
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            val_loss += criterion(outputs, targets)
            
            topv,topi = outputs.topk(1, dim=1)
            targets = targets.unsqueeze(-1)
            val_accuracy += float((topi == targets).sum())/targets.shape[0]
            #pdb.set_trace()
            # Print log info
            if i % args.log_step == 0:
                print('step {}/{}, time {}, accuracy {}, bleu_score {}{}{}{}'.format(i, len(data_loader), timeSince(start), val_accuracy/(i+1), sb1/(i+1),sb2/(i+1),sb3/(i+1),sb4/(i+1)))   
            
            sentence_length=0
            for j in range(len(lengths)):
                candidate = [vocab.idx2word[int(idx[0])] for idx in topi[sentence_length:sentence_length+lengths[j]]]
                reference = [[vocab.idx2word[int(idx[0])] for idx in targets[sentence_length:sentence_length+lengths[j]]]]
                #print(cadidate)
                #pdb.set_trace()
                sb1+=float(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))/len(lengths)
                sb2+=float(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))/len(lengths)
                sb3+=float(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))/len(lengths)
                sb4+=float(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))/len(lengths)
                sentence_length+=lengths[j]
    val_loss = val_loss/len(data_loader)
    print('val_loss = {:.3f}'.format(val_loss))

    #reference = [['this', 'is', 'small', 'test']]
    #candidate = ['this', 'is', 'a', 'test']
    print('Cumulative 1-gram: %f' % (sb1/len(data_loader)))
    print('Cumulative 2-gram: %f' % (sb2/len(data_loader)))
    print('Cumulative 3-gram: %f' % (sb3/len(data_loader)))
    print('Cumulative 4-gram: %f' % (sb4/len(data_loader)))
    print('accuracy: ',val_accuracy/len(data_loader))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default= home+ '/data2/models/encoder-3-1000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default= home+ '/data2/models/decoder-3-1000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default= home+ '/data2/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default= home+ '/data2/val_resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default= home+ '/data2/annotations/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=50, help='step size for prining log info')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=128, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    print(args)
    val(args)
    
    
    
    
    
    
    
    
    
   
    