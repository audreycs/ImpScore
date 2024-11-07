from torch import optim
from model import ImpModel
from load_dataset import LoadDataset
import argparse
import time
from utils import *
import os
import json
import tqdm
import csv
import random


def generate_train_batches(train_pairs, batch_size, shuffle=True):
    """
    Generate batches from sentence pairs
    :param sentence_pairs: list of pairs (pos_pair, neg_pair)
    :param batch_size: int
    :param shuffle: bool
    :return: list
    """
    if shuffle:
        np.random.shuffle(train_pairs)

    batches = []
    for i in range(0, len(train_pairs), batch_size):
        batch = train_pairs[i:i + batch_size]
        batches.append(batch)
    return batches


def generate_valid_test_batches(sentence_pairs, batch_size, shuffle=True):
    """
    Generate batches from sentence pairs
    :param sentence_pairs: list of sentence pairs
    :param batch_size: int
    :param shuffle: bool
    :return: list
    """
    if shuffle:
        np.random.shuffle(sentence_pairs)

    batches = []
    for i in range(0, len(sentence_pairs), batch_size):
        batch = sentence_pairs[i:i + batch_size]
        batches.append(batch)
    return batches


def avg_pragmatic_result(prag_label_list, prag_distance_list):
    """
    Calculate the average pragmatic distance for positive and negative samples
    :param prag_label_list:
    :param prag_distance_list:
    :return:
    """
    prag_label_list = [int(item) for item in prag_label_list]
    prag_distance_list = [float(item) for item in prag_distance_list]

    assert len(prag_label_list) == len(
        prag_distance_list), "Length mismatch between prag_label_list and prag_score_list"

    # get index list of prag_label_list where value is 1
    pos_index = [i for i, item in enumerate(prag_label_list) if item == 1]
    neg_index = [i for i, item in enumerate(prag_label_list) if item == 0]

    # get the corresponding prag_score_list
    pos_dis = [prag_distance_list[i] for i in pos_index]
    neg_dis = [prag_distance_list[i] for i in neg_index]

    # calculate the average score for positive and negative samples
    avg_pos_dis = sum(pos_dis) / len(pos_dis)
    avg_neg_dis = sum(neg_dis) / len(neg_dis)

    return avg_pos_dis, avg_neg_dis


def write_test_result(args, program_start_time, test_pairs, imp_scores1_list, imp_scores2_list,
                      prag_scores_list, prag_labels_list):
    file_name = f"test_{program_start_time}.csv"
    file_path = os.path.join(args.test_result_dir, file_name)
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["implicit", "explicit", "imp_score1", "imp_score2", "prag_distance", "prag_label"])
        assert len(test_pairs) == len(imp_scores1_list) == len(imp_scores2_list) == len(prag_scores_list) == len(
            prag_labels_list), "Length mismatch"
        for pair, score1, score2, prag_score, prag_label in zip(test_pairs, imp_scores1_list, imp_scores2_list,
                                                                prag_scores_list, prag_labels_list):
            writer.writerow([pair[0], pair[1], score1, score2, prag_score, prag_label])


def analyze_prag_result(prag_distances_list):
    assert len(prag_distances_list) % 2 == 0, "Length of prag_scores_list should be even"

    pos_distances = prag_distances_list[::2]
    neg_distances = prag_distances_list[1::2]

    avg_pos_distance = sum(pos_distances) / len(pos_distances)
    avg_neg_distance = sum(neg_distances) / len(neg_distances)

    pair_dis = zip(pos_distances, neg_distances)

    correct_count = 0
    for pos_dis, neg_dis in pair_dis:
        if pos_dis < neg_dis:
            correct_count += 1

    return correct_count / len(pos_distances), avg_pos_distance, avg_neg_distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implicitness')

    real_start_time = time.time()

    # arguments for file path
    parser.add_argument("--data_dir", type=str, default='./', help='Folder path of training data')
    parser.add_argument('--model_dir', type=str, default='./saved_models/', help='folder to save models')
    parser.add_argument("--fig_dir", type=str, default='./figs/', help='path to save figures')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='folder to save logs')
    parser.add_argument('--test_result_dir', type=str, default='./test_results/', help='folder to save logs')
    parser.add_argument('--delete_model', action='store_true', help='delete the saved model after training')
    parser.add_argument('--delete_figure', action='store_true', help='delete the generated figures after training')

    # arguments for training
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs for training')
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=2**13, help='batch size for training')
    parser.add_argument("--wd", type=float, default=0.001, help='weight decay for optimizer')
    parser.add_argument("--valid_split", type=float, default=0.1, help="validation split ratio")
    parser.add_argument("--valid_freq", type=int, default=1, help="the epoch frequency of validation")
    parser.add_argument('--seed', type=int, default=42, help='global random seed (used for splitting data)')

    # arguments for model
    parser.add_argument('--feat_dim', type=int, default=64, help='dimension of the feature space')
    parser.add_argument('--margin1', type=float, default=0.5, help='gap of implicitness score')
    parser.add_argument('--margin2', type=float, default=0.7, help='gap of pragmatic distance')
    parser.add_argument('--alpha', type=float, default=1.0, help='proportion of the pragmatic loss')
    parser.add_argument('--space_direct', type=str, default='p2s', help="choose from 'p2s', 's2p', or 'another'")
    parser.add_argument('--imp_metric', type=str, default='cos', help="metric for computing the implicitness score, choose from \"euc\" or \"cos\"")
    parser.add_argument('--prag_metric', type=str, default='euc', help="metric for computing the pragmatic distance, choose from \"euc\" or \"cos\"")

    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)

    program_start_time = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))

    set_logging(args, "Implicitness")
    show_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args_dict = vars(args)
    # add device to args_dict
    args_dict['device'] = device

    data_loader = LoadDataset(args.data_dir)
    train_pairs = data_loader.train_pairs
    valid_pairs = data_loader.valid_pairs
    test_pairs = data_loader.test_pairs

    model = ImpModel(args, device)
    model.to(device)
    model.print_model_size()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    epoch_loss_list = []
    valid_loss_list = []
    valid_prag_distance_list = []
    good_pred_list = []
    best_performance = 0.0
    best_model_saved_file = f"model_state_dict_{program_start_time}.pth"
    if args.mode == 'train':
        logging.info("\n--------------start training--------------")
        for epoch in range(args.epoch):
            time1 = time.time()
            train_loss = 0
            # convert train_pairs to batches
            train_batches = generate_train_batches(train_pairs, batch_size=args.batch_size, shuffle=True)
            model.train()
            for batch_index, pairs in tqdm.tqdm(enumerate(train_batches), total=len(train_batches)):
                pos_pairs = pairs[:, 0]
                neg_pairs = pairs[:, 1]

                optimizer.zero_grad()
                time_temp = time.time()
                loss = model(pos_pairs, neg_pairs)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            time2 = time.time()
            logging.info(f"[epoch {epoch + 1}/{args.epoch}]  "
                         f"epoch_loss = {train_loss:.4f},  "
                         # f"per_sample_loss = {total_loss/len(train_pairs):.4f},  "
                         f"runtime = {(time2 - time1):.4f}s")
            epoch_loss_list.append((epoch, train_loss))

            model.eval()
            with torch.no_grad():
                if (epoch + 1) % args.valid_freq == 0:
                    good_pred = 0
                    # convert test_pairs to batches
                    valid_batches = generate_valid_test_batches(valid_pairs, batch_size=args.batch_size, shuffle=False)
                    valid_loss = 0

                    valid_score1_list = []
                    valid_score2_list = []
                    valid_prag_labels_list = []
                    valid_prag_distances_list = []

                    for pairs in valid_batches:
                        sentences1 = pairs[:, 0]
                        sentences2 = pairs[:, 1]
                        pragmatic_labels = pairs[:, 2]
                        valid_prag_labels_list.extend(pragmatic_labels)
                        imp_loss, imp_scores1, imp_scores2, prag_distances = model.test(sentences1, sentences2)
                        valid_loss += imp_loss.item()
                        prag_distances = [p.item() for p in prag_distances]
                        valid_prag_distances_list.extend(prag_distances)
                        for score1, score2 in zip(imp_scores1, imp_scores2):
                            valid_score1_list.append(score1.item())
                            valid_score2_list.append(score2.item())
                            if score1 > score2:
                                good_pred += 1
                    accuracy = good_pred / len(valid_pairs)
                    avg_pos_score, avg_neg_score = avg_pragmatic_result(valid_prag_labels_list, valid_prag_distances_list)
                    prag_acc, avg_pos_distance, avg_neg_distance = analyze_prag_result(valid_prag_distances_list)
                    logging.info(f"[validation] loss = {valid_loss:.4f},  implicitness_accuracy = {accuracy:.4f},  "
                                 f"avg_pos_prag_dis = {avg_pos_score:.4f},  avg_neg_prag_dis = {avg_neg_score:.4f},  "
                                 f"pragmatic_accuracy = {prag_acc:.4f}")
                    valid_loss_list.append((epoch, valid_loss))
                    good_pred_list.append(accuracy)
                    valid_prag_distance_list.append(avg_neg_distance-avg_pos_distance)

                    if accuracy > best_performance:
                        best_performance = accuracy

                        # Save the dictionary to a JSON file
                        with open(os.path.join(args.model_dir, f"model_args_{program_start_time}.json"), 'w') as f:
                            json.dump(args_dict, f)

                        torch.save(model.state_dict(), os.path.join(args.model_dir, best_model_saved_file))
                        logging.info(
                            f"[Best performance so far. Model saved to \"{args.model_dir}\"{best_model_saved_file}]")

    logging.info("\n--------------start testing--------------")
    # load the best model
    if args.mode == 'train':
        model.load_state_dict(torch.load(os.path.join(args.model_dir, best_model_saved_file)))
    else:
        saved_model = 'model_state_dict_07-15-21-20-57.pth'
        try:
            model.load_state_dict(torch.load(os.path.join(args.model_dir, saved_model)))
        except FileNotFoundError:
            logging.error(f"model {saved_model} not found")
            exit(1)
    model.eval()

    with torch.no_grad():
        good_pred = 0
        # convert test_pairs to batches
        test_batches = generate_valid_test_batches(test_pairs, batch_size=args.batch_size, shuffle=False)
        test_loss = 0
        test_imp_scores1_list, test_imp_scores2_list = [], []
        test_prag_distances_list = []
        test_prag_labels_list = []
        for pairs in test_batches:
            sentences1 = pairs[:, 0]
            sentences2 = pairs[:, 1]
            pragmatic_labels = pairs[:, 2]
            imp_loss, imp_scores1, imp_scores2, prag_distances = model.test(sentences1, sentences2)
            test_loss += imp_loss.item()
            test_prag_labels_list.extend(pragmatic_labels)
            for distance in prag_distances:
                test_prag_distances_list.append(distance.item())
            for score1, score2 in zip(imp_scores1, imp_scores2):
                test_imp_scores1_list.append(score1.item())
                test_imp_scores2_list.append(score2.item())
                if score1 > score2:
                    good_pred += 1
        logging.info(f"[testing] loss = {test_loss:.4f}")
        accuracy = good_pred / len(test_pairs)
        logging.info(f"[testing] implicitness_accuracy = {accuracy:.4f}")
        avg_pos_score, avg_neg_score = avg_pragmatic_result(test_prag_labels_list, test_prag_distances_list)
        avg_imp_score_1 = sum(test_imp_scores1_list) / len(test_imp_scores1_list)
        avg_imp_score_2 = sum(test_imp_scores2_list) / len(test_imp_scores2_list)
        write_test_result(args,
                          program_start_time,
                          test_pairs,
                          test_imp_scores1_list,
                          test_imp_scores2_list,
                          test_prag_distances_list,
                          test_prag_labels_list)
        prag_acc, _, _ = analyze_prag_result(test_prag_distances_list)
        logging.info(f"[testing] avg_pos_prag_dis = {avg_pos_score:.4f},  "
                     f"avg_imp_score_1 = {avg_imp_score_1:.4f},  "
                     f"avg_imp_score_2 = {avg_imp_score_2:.4f},  "
                     f"avg_neg_prag_dis = {avg_neg_score:.4f},  "
                     f"pragmatic_accuracy = {prag_acc:.4f}")

    logging.info("")

    if args.mode == 'train':
        plot_loss_acc(args, epoch_loss_list, valid_loss_list, good_pred_list, valid_prag_distance_list, program_start_time)
        plot_scores(args, valid_score1_list, valid_score2_list, program_start_time, point_size=100)
    plot_violin_distribution(args,
                             program_start_time,
                             test_imp_scores1_list,
                             test_imp_scores2_list,
                             test_prag_distances_list,
                             test_prag_labels_list)

    real_end_time = time.time()
    logging.info("")
    logging.info(f"Program runtime: {real_end_time - real_start_time:.4f}s")

    if args.delete_model:
        os.remove(os.path.join(args.model_dir, best_model_saved_file))
        best_model_args_file = f"model_args_{program_start_time}.json"
        os.remove(os.path.join(args.model_dir, best_model_args_file))
        logging.info(f"Model files of \"{program_start_time}\" deleted")

    if args.delete_figure:
        fig_file_list = os.listdir(args.fig_dir)
        # if the figure file name contain 'program_start_time', delete it
        for fig_file in fig_file_list:
            if program_start_time in fig_file:
                os.remove(os.path.join(args.fig_dir, fig_file))
        logging.info(f"Figure files of \"{program_start_time}\" deleted")
