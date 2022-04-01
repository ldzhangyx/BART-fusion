import torch
from tqdm import tqdm
from data import LyricsCommentsDatasetPsuedo
from torch import utils, nn
from model import CommentGenerator
import transformers
import time
import statistics
import os
import random
import datasets

IS_LOAD = False
LOAD_EPOCH = 0
EPOCH = 20
BATCH_SIZE = 8
LOG_INTERVAL = 100
SAMPLE_INTERVAL = 2000
VALIDATION_INTERVAL = 2
LOG_FOLDER = "log/"
MODEL_FOLDER = "model/"
EARLY_STOPPING_INTERVAL = 5
MODEL_NAME = "bart_baseline_full_256"
CHOICE_NUMBER = 5
DATASET_PATH = "dataset_not_negative_256.pkl"

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

dataset = LyricsCommentsDatasetPsuedo(dataset_path=DATASET_PATH)
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
valid_dataloader = utils.data.DataLoader(valid_dataset,
                                         batch_size=32,
                                         shuffle=False)

model = CommentGenerator().cuda()

criterion = nn.CrossEntropyLoss()

optimizer = transformers.Adafactor(model.parameters(), warmup_init=False, relative_step=False,
                                   lr=6e-4,
                                   )

loss_stat = list()
start_time = time.time()
start_time_local = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

early_stop_token = (0.0, 0)

model.train()
for epoch in range(1 + LOAD_EPOCH, EPOCH + 1 + LOAD_EPOCH):
    for batch_index, [lyrics, comment] in enumerate(train_dataloader):
        # pre-process data
        input_sentences = lyrics
        raw_labels = comment
        output = model(input_sentences, raw_labels)
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
                  f"Loss: {statistics.mean(loss_stat[-1 * BATCH_SIZE:])}\t" \
                  f"Avg loss: {statistics.mean(loss_stat)}"
            if __debug__:
                print(log)
            with open(os.path.join(LOG_FOLDER, MODEL_NAME + "_" + start_time_local + ".txt"), 'a+', encoding='utf-8') as r:
                r.write(log)
                r.write("\n")
            loss_stat = list()

        if batch_index and batch_index % SAMPLE_INTERVAL == 0:

            model.eval()
            samples_list = random.choices(valid_dataset, k=CHOICE_NUMBER)
            sample_sentence, sample_label = zip(*samples_list)
            output_samples = model.generate(sample_sentence)
            for sample_index in range(CHOICE_NUMBER):
                log = f"Lyrics: {sample_sentence[sample_index]}\n" \
                      f"Sample outputs: {output_samples[sample_index]}\n" \
                      f"Ground Truth: {sample_label[sample_index]}"
                if __debug__:
                    print(log)
                with open(os.path.join(LOG_FOLDER, MODEL_NAME + "_" + start_time_local + ".txt"), 'a+', encoding='utf-8') as r:
                    r.write(log)
                    r.write("\n")
            model.train()

    if epoch and epoch % VALIDATION_INTERVAL == 0:
        model.eval()
        metrics = datasets.load_metric('rouge')
        valid_dataloader = utils.data.DataLoader(valid_dataset,
                                                 batch_size=32,
                                                 shuffle=False)
        for batch_index_valid, [lyrics_valid, comment_valid] in enumerate(valid_dataloader):
            output_samples = model.generate(lyrics_valid)
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

        if epoch:
            torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, f"{MODEL_NAME}_epoch{epoch}.pt"))
            torch.save(optimizer.state_dict(),
                       os.path.join(MODEL_FOLDER, f"{MODEL_NAME}_optim_epoch{epoch}.pt"))

        # early stopping
        if score['rouge1'].mid.recall <= early_stop_token[0] and epoch > (
                early_stop_token[1] + EARLY_STOPPING_INTERVAL):
            print(f"Early Stopping. Best Score: {early_stop_token[0]} at Epoch {early_stop_token[1]}.")

        model.train()