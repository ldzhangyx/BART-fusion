import torch
from tqdm import tqdm
from data import LyricsCommentsDatasetPsuedo_fusion
from torch import utils, nn
from model_fusion import CommentGenerator_fusion
import transformers
import time
import statistics
import os
import random
import datasets

IS_LOAD = False
LOAD_EPOCH = 0
EPOCH = 50
BATCH_SIZE = 8
LOG_INTERVAL = 100
SAMPLE_INTERVAL = 1000
VALIDATION_INTERVAL = 2
LOG_FOLDER = "log/"
MODEL_FOLDER = "model/"
SAVE_INTERVAL = 2
EARLY_STOPPING_INTERVAL = 5
MODEL_NAME = "bart_fusion_full_256"
CHOICE_NUMBER = 2
DATASET_PATH = "/homes/yz007/multimodal-transformer/comment_generator/dataset_full_256.pkl"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

dataset = LyricsCommentsDatasetPsuedo_fusion(dataset_path=DATASET_PATH)
dataset_length = len(dataset)

train_dataset_length = int(dataset_length * 0.9)
valid_dataset_length = dataset_length - train_dataset_length
train_dataset, valid_dataset = utils.data.random_split(dataset,
                                                       [train_dataset_length,
                                                        valid_dataset_length],
                                                       generator=torch.Generator().manual_seed(42))
train_dataloader = utils.data.DataLoader(train_dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)
# valid_dataloader = utils.data.DataLoader(valid_dataset,
#                                          batch_size=32,
#                                          shuffle=False)

model = CommentGenerator_fusion().cuda()

criterion = nn.CrossEntropyLoss()



# optimizer = transformers.Adafactor(filter(lambda p: p.requires_grad, model.parameters()),
#                                    lr=6e-4,
#                                    )
optimizer = transformers.Adafactor(model.parameters(), warmup_init=False, relative_step=False,
                                   lr=6e-4,
                                   )

if IS_LOAD:
    model.load_state_dict(torch.load("/homes/yz007/multimodal-transformer/comment_generator/model/bart_fusion_positive_256_6e-4_epoch6.pt"))
    optimizer.load_state_dict(torch.load("/homes/yz007/multimodal-transformer/comment_generator/model/bart_fusion_positive_256_6e-4_optim_epoch6.pt"))

loss_stat = list()
start_time = time.time()
start_time_local = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

early_stop_token = [0.0, 0]
validation_loss_history = list()

model.train()
for epoch in range(1 + LOAD_EPOCH, EPOCH + 1 + LOAD_EPOCH):
    for batch_index, [lyrics, comment, music_id] in enumerate(train_dataloader):
        # pre-process data
        input_sentences = lyrics
        raw_labels = comment
        output = model(input_sentences, music_id, raw_labels)
        loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_stat.append(loss.item())

        # log
        if batch_index and batch_index % LOG_INTERVAL == 0:
            curr_time = time.time()
            passed_time_all = curr_time - start_time
            time_str = f"{int(passed_time_all / 60)}:{int(passed_time_all % 60)}"
            log = f"{MODEL_NAME}\t" \
                  f"Time: {time_str}\t" \
                  f"Epoch {epoch}: {batch_index}/{int(len(train_dataloader.dataset) / BATCH_SIZE)}\t" \
                  f"Loss: {statistics.mean(loss_stat[-1 * LOG_INTERVAL * BATCH_SIZE:])}\t" \
                  f"Avg loss: {statistics.mean(loss_stat)}"
            if __debug__:
                print(log)
            with open(os.path.join(LOG_FOLDER, MODEL_NAME + '_' + start_time_local + ".txt"), 'a+',
                      encoding='utf-8') as r:
                r.write(log)
                r.write("\n")
            loss_stat = list()

        if batch_index and batch_index % SAMPLE_INTERVAL == 0:
            # make samples
            model.eval()
            samples_list = random.choices(valid_dataset, k=CHOICE_NUMBER)
            sample_sentence, sample_label, music_ids = zip(*samples_list)
            with torch.no_grad():
                output_samples = model.generate(sample_sentence, music_ids)
            for sample_index in range(CHOICE_NUMBER):
                log = f"Lyrics: {sample_sentence[sample_index]}\n" \
                      f"Sample outputs: {output_samples[sample_index]}\n" \
                      f"Ground Truth: {sample_label[sample_index]}"
                if __debug__:
                    print(log)
                with open(os.path.join(LOG_FOLDER, MODEL_NAME + '_' + start_time_local + ".txt"), 'a+',
                          encoding='utf-8') as r:
                    r.write(log)
                    r.write("\n")

            # validation loss
            valid_dataloader = utils.data.DataLoader(valid_dataset,
                                                     batch_size=8,
                                                     shuffle=False)
            valid_loss_stat = list()
            for batch_index_valid, [lyrics_valid, comment_valid, music_id_valid] in enumerate(valid_dataloader):
                with torch.no_grad():
                    output_valid = model(lyrics_valid, music_id_valid, comment_valid)
                valid_loss = output_valid.loss.item()
                valid_loss_stat.append(valid_loss)
                if batch_index_valid > 15:
                    break
            valid_loss_mean = statistics.mean(valid_loss_stat)
            validation_loss_history.append(valid_loss_mean)
            log = f"{MODEL_NAME}\t" \
                  f"Time: {time_str}\t" \
                  f"Epoch {epoch}: {batch_index}/{int(len(train_dataloader.dataset) / BATCH_SIZE)}\t" \
                  f"Validation Loss: {valid_loss_mean}\t"
            if __debug__:
                print(log)
            with open(os.path.join(LOG_FOLDER, MODEL_NAME + '_' + start_time_local + ".txt"), 'a+',
                      encoding='utf-8') as r:
                r.write(log)
                r.write("\n")

            # back to train
            model.train()

    if epoch and epoch % VALIDATION_INTERVAL == 0:
        model.eval()
        metrics = datasets.load_metric('rouge')
        valid_dataloader = utils.data.DataLoader(valid_dataset,
                                                 batch_size=8,
                                                 shuffle=False)
        for batch_index_valid, [lyrics_valid, comment_valid, music_id_valid] in enumerate(valid_dataloader):
            with torch.no_grad():
                output_samples = model.generate(lyrics_valid, music_id_valid)
            metrics.add_batch(predictions=output_samples, references=comment_valid)
            # control time.
            if batch_index_valid > 10:
                break
        score = metrics.compute()
        if __debug__:
            print(str(score))
        with open(os.path.join(LOG_FOLDER, MODEL_NAME + '_' + start_time_local + ".txt"), 'a+',
                  encoding='utf-8') as r:
            r.write(str(score))
            r.write("\n")

        # save
        if score['rouge1'].mid.recall > early_stop_token[0]:
            early_stop_token = [score['rouge1'].mid.recall, epoch]  # replace to the best
            torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, f"{MODEL_NAME}_best.pt"))
            torch.save(optimizer.state_dict(),
                       os.path.join(MODEL_FOLDER, f"{MODEL_NAME}_optim_best.pt"))

        # save
        if epoch and epoch % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, f"{MODEL_NAME}_epoch{epoch}.pt"))
            torch.save(optimizer.state_dict(),
                       os.path.join(MODEL_FOLDER, f"{MODEL_NAME}_optim_epoch{epoch}.pt"))

        # early stopping
        if len(validation_loss_history) > EARLY_STOPPING_INTERVAL:
            if min(validation_loss_history[-2 * EARLY_STOPPING_INTERVAL:]) == validation_loss_history[-2 * EARLY_STOPPING_INTERVAL]:
                print(f"Early Stopping. Best Score: {early_stop_token[0]} at Epoch {early_stop_token[1]}.")
                break
        if score['rouge1'].mid.recall <= early_stop_token[0] and epoch > (
                early_stop_token[1] + EARLY_STOPPING_INTERVAL):
            print(f"Early Stopping. Best Score: {early_stop_token[0]} at Epoch {early_stop_token[1]}.")
            break
        model.train()

print(f"Training Complete. Best Score: {early_stop_token[0]} at Epoch {early_stop_token[1]}.")
