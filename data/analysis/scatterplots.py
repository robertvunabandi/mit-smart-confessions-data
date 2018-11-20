import constants
import data.analysis.analysis_util as a_util


# adding comments to the number of reactions
N_REACTIONS = len(constants.FB_REACTIONS) + 1
REACTION_NAMES = ([r.lower() for r in constants.FB_REACTIONS] + ["comment"])


def plot_char_length_to_rxn_scatters(max_char_length: int = None, save: bool = False) -> None:
    # do char length and reaction  count
    all_char_lengths = [len(t) for t in a_util.get_data_texts()]
    all_rxn_labels = a_util.get_data_labels()
    label_lists = []
    for index in range(N_REACTIONS):
        char_lengths, rxn_labels = [], []
        for char_length, label in zip(all_char_lengths, all_rxn_labels):
            if max_char_length is None or char_length <= max_char_length:
                char_lengths.append(char_length)
                rxn_labels.append(label[index])
        label_lists.append((char_lengths, rxn_labels))
    for reaction_name, bundle in zip(REACTION_NAMES, label_lists):
        char_lengths, rxn_labels = bundle
        scatter = a_util.Scatterplot(char_lengths, rxn_labels)
        title = "Scatter for char-count (X) and %ss (Y)" % reaction_name
        xlabel = "character count"
        ylabel = "%s count" % reaction_name
        fname = "char_count_v_%s" % reaction_name
        if max_char_length is not None:
            title = "%s %s" % (title, "(max char length %d)" % max_char_length)
            fname = "%s %s" % (fname, "mlen%d" % max_char_length)
        scatter.plot(title, xlabel, ylabel, save=save, fname=fname)


def plot_rxn_to_rxn_scatters(save: bool = False) -> None:
    all_rxn_labels = a_util.get_data_labels()
    reaction_lists = [
        [all_rxn_labels[i][index] for i in range(len(all_rxn_labels))]
        for index in range(N_REACTIONS)
    ]
    for r1index in range(N_REACTIONS - 1):
        for r2index in range(r1index + 1, N_REACTIONS):
            r1name, r2name = REACTION_NAMES[r1index], REACTION_NAMES[r2index]
            scatter = a_util.Scatterplot(reaction_lists[r1index], reaction_lists[r2index])
            title = "Scatter for %ss to %ss" % (r1name, r2name)
            xlabel = "%s count" % r1name
            ylabel = "%s count" % r2name
            fname = "%s_v_%s" % (r1name, r2name)
            scatter.plot(title, xlabel, ylabel, save=save, fname=fname)


if __name__ == '__main__':
    # plot_char_length_to_rxn_scatters(max_char_length=1000, save=True)
    plot_rxn_to_rxn_scatters(save=True)
