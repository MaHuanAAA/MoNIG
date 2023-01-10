from torch import nn
from src import models
from src.utils import *
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.eval_metrics import *


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)  # if num of labels is 1

            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk

            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 16 else model
            if hyp_params.model == 'BASE':
                if batch_chunk > 1:
                    raw_loss = combined_loss = 0
                    text_chunks = text.chunk(batch_chunk, dim=0)
                    audio_chunks = audio.chunk(batch_chunk, dim=0)
                    vision_chunks = vision.chunk(batch_chunk, dim=0)
                    eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)

                    for i in range(batch_chunk):
                        text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        preds_i, hiddens_i = net(text_i, audio_i, vision_i)

                        if hyp_params.dataset == 'iemocap':
                            preds_i = preds_i.view(-1, 2)
                            eval_attr_i = eval_attr_i.view(-1)
                        raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                        raw_loss_i.backward()
                        raw_loss = raw_loss + raw_loss_i
                    combined_loss = raw_loss
                else:
                    preds, hiddens = net(text, audio, vision)
                    if hyp_params.dataset == 'iemocap':
                        preds = preds.view(-1, 2)
                        eval_attr = eval_attr.view(-1)
                    raw_loss = criterion(preds, eval_attr)
                    combined_loss = raw_loss
                    combined_loss.backward()
            elif hyp_params.model == 'MOE':
                if batch_chunk > 1:
                    raw_loss = combined_loss = 0
                    text_chunks = text.chunk(batch_chunk, dim=0)
                    audio_chunks = audio.chunk(batch_chunk, dim=0)
                    vision_chunks = vision.chunk(batch_chunk, dim=0)
                    eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)

                    for i in range(batch_chunk):
                        text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i, mu_v_i, v_v_i, \
                        alpha_v_i, beta_v_i = net(text_i, audio_i, vision_i)
                        mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                            moe_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i)
                        mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                            moe_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, mu_v_i, v_v_i, alpha_v_i, beta_v_i)
                        if hyp_params.dataset == 'iemocap':
                            eval_attr_i = eval_attr_i.view(-1)
                        raw_loss_i = criterion_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, eval_attr_i, hyp_params) + \
                                     criterion_nig(mu_a_i, v_a_i, alpha_a_i, beta_a_i, eval_attr_i, hyp_params) + \
                                     criterion_nig(mu_v_i, v_v_i, alpha_v_i, beta_v_i, eval_attr_i, hyp_params) + \
                                     criterion_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, eval_attr_i, hyp_params)

                        raw_loss_i.backward()
                        raw_loss = raw_loss + raw_loss_i
                    combined_loss = raw_loss

                else:
                    mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i, mu_v_i, v_v_i, \
                    alpha_v_i, beta_v_i = net(text, audio, vision)
                    mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i)
                    mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, mu_v_i, v_v_i, alpha_v_i, beta_v_i)
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.view(-1)
                    raw_loss = criterion_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, eval_attr, hyp_params) + \
                                 criterion_nig(mu_a_i, v_a_i, alpha_a_i, beta_a_i, eval_attr, hyp_params) + \
                                 criterion_nig(mu_v_i, v_v_i, alpha_v_i, beta_v_i, eval_attr, hyp_params) + \
                                 criterion_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, eval_attr, hyp_params)
                    combined_loss = raw_loss
                    combined_loss.backward()
            elif hyp_params.model == 'Comb':
                if batch_chunk > 1:
                    raw_loss = combined_loss = 0
                    text_chunks = text.chunk(batch_chunk, dim=0)
                    audio_chunks = audio.chunk(batch_chunk, dim=0)
                    vision_chunks = vision.chunk(batch_chunk, dim=0)
                    eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)

                    for i in range(batch_chunk):
                        text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i, mu_v_i, v_v_i, \
                        alpha_v_i, beta_v_i, mu_c_i, v_c_i, alpha_c_i, beta_c_i = net(text_i, audio_i, vision_i)
                        mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                            moe_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i)
                        mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                            moe_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, mu_v_i, v_v_i, alpha_v_i, beta_v_i)
                        mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                            moe_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, mu_c_i, v_c_i, alpha_c_i, beta_c_i)
                        if hyp_params.dataset == 'iemocap':
                            eval_attr_i = eval_attr_i.view(-1)
                        raw_loss_i = criterion_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, eval_attr_i, hyp_params) + \
                                     criterion_nig(mu_a_i, v_a_i, alpha_a_i, beta_a_i, eval_attr_i, hyp_params) + \
                                     criterion_nig(mu_v_i, v_v_i, alpha_v_i, beta_v_i, eval_attr_i, hyp_params) + \
                                     criterion_nig(mu_c_i, v_c_i, alpha_c_i, beta_c_i, eval_attr_i, hyp_params) + \
                                     criterion_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, eval_attr_i, hyp_params)
                        raw_loss_i.backward()
                        raw_loss = raw_loss + raw_loss_i
                    combined_loss = raw_loss
                else:
                    mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i, mu_v_i, v_v_i, \
                    alpha_v_i, beta_v_i, mu_c_i, v_c_i, alpha_c_i, beta_c_i = net(text, audio, vision)
                    mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i)
                    mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, mu_v_i, v_v_i, alpha_v_i, beta_v_i)
                    mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, mu_c_i, v_c_i, alpha_c_i, beta_c_i)
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.view(-1)
                    raw_loss = criterion_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, eval_attr, hyp_params) + \
                                 criterion_nig(mu_a_i, v_a_i, alpha_a_i, beta_a_i, eval_attr, hyp_params) + \
                                 criterion_nig(mu_v_i, v_v_i, alpha_v_i, beta_v_i, eval_attr, hyp_params) + \
                                 criterion_nig(mu_c_i, v_c_i, alpha_c_i, beta_c_i, eval_attr, hyp_params) + \
                                 criterion_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, eval_attr, hyp_params)
                    combined_loss = raw_loss
                    combined_loss.backward()
            elif hyp_params.model == 'NIG':
                if batch_chunk > 1:
                    raw_loss = combined_loss = 0
                    text_chunks = text.chunk(batch_chunk, dim=0)
                    audio_chunks = audio.chunk(batch_chunk, dim=0)
                    vision_chunks = vision.chunk(batch_chunk, dim=0)
                    eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)

                    for i in range(batch_chunk):
                        text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        mu_i, v_i, alpha_i, beta_i = net(text_i, audio_i, vision_i)

                        if hyp_params.dataset == 'iemocap':
                            eval_attr_i = eval_attr_i.view(-1)
                        raw_loss_i = criterion_nig(mu_i, v_i, alpha_i, beta_i, eval_attr_i, hyp_params)
                        raw_loss_i.backward()
                        raw_loss = raw_loss + raw_loss_i
                    combined_loss = raw_loss
                else:
                    mu_i, v_i, alpha_i, beta_i = net(text, audio, vision)
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.view(-1)
                    raw_loss = criterion_nig(mu_i, v_i, alpha_i, beta_i, eval_attr, hyp_params)
                    combined_loss = raw_loss
                    combined_loss.backward()

            elif hyp_params.model == 'GAUSSIAN':
                if batch_chunk > 1:
                    raw_loss = combined_loss = 0
                    text_chunks = text.chunk(batch_chunk, dim=0)
                    audio_chunks = audio.chunk(batch_chunk, dim=0)
                    vision_chunks = vision.chunk(batch_chunk, dim=0)
                    eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)

                    for i in range(batch_chunk):
                        text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        mu_i, sigma_i = net(text_i, audio_i, vision_i)

                        if hyp_params.dataset == 'iemocap':
                            eval_attr_i = eval_attr_i.view(-1)
                        raw_loss_i = sum(0.5 * (torch.log(2 * np.pi * (sigma_i)) + ((eval_attr_i - mu_i) ** 2) / (sigma_i))) / len(mu_i)
                        raw_loss = raw_loss + raw_loss_i
                        raw_loss_i.backward()
                    combined_loss = raw_loss
                else:
                    mu, sigma = net(text, audio, vision)
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.view(-1)
                    raw_loss = sum(0.5 * (torch.log(2 * np.pi * (sigma)) + ((eval_attr - mu) ** 2) / (sigma))) / len(mu)
                    combined_loss = raw_loss
                    combined_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss = proc_loss + raw_loss.item() * batch_size
            proc_size = proc_size + batch_size
            epoch_loss = epoch_loss + combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()

        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()

                batch_size = text.size(0)

                net = nn.DataParallel(model) if batch_size > 16 else model
                if hyp_params.model == 'MOE':
                    mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i, mu_v_i, v_v_i, \
                    alpha_v_i, beta_v_i = net(text, audio, vision)
                    mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i)
                    preds, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, mu_v_i, v_v_i, alpha_v_i, beta_v_i)
                if hyp_params.model == 'Comb':
                    mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i, mu_v_i, v_v_i, \
                    alpha_v_i, beta_v_i, mu_c_i, v_c_i, alpha_c_i, beta_c_i = net(text, audio, vision)
                    mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_l_i, v_l_i, alpha_l_i, beta_l_i, mu_a_i, v_a_i, alpha_a_i, beta_a_i)
                    mu_s_i, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, mu_v_i, v_v_i, alpha_v_i, beta_v_i)
                    preds, v_s_i, alpha_s_i, beta_s_i = \
                        moe_nig(mu_s_i, v_s_i, alpha_s_i, beta_s_i, mu_c_i, v_c_i, alpha_c_i, beta_c_i)
                if hyp_params.model == 'NIG':
                    preds, _, _, _ = net(text, audio, vision)
                if hyp_params.model == 'GAUSSIAN':
                    preds, _ = net(text, audio, vision)
                if hyp_params.dataset == 'iemocap':
                    eval_attr = eval_attr.view(-1)
                total_loss = total_loss + criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 50)
        print(
            'Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss,
                                                                                             test_loss))
        print("-" * 50)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, criterion, test=True)
    acc7, acc2, f_score, mae, corr = 0, 0, 0, 0, 0
    if hyp_params.dataset == "mosei_senti":
        acc7, acc2, f_score, mae, corr = eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        acc7, acc2, f_score, mae, corr = eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)
    return acc7, acc2, f_score, mae, corr

    # sys.stdout.flush()
    # input('[Press Any Key to start another run]')
