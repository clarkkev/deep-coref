import directories
import utils
from collections import defaultdict
from operator import itemgetter


ENDC = '\033[0m'
COLORS = ['\033[' + str(n) + 'm' for n in list(range(91, 97)) + [90]]


def subscript(n):
    return "".join([unichr(8320 + int(d)) for d in str(n)])


def write_links(model_path, dataset_name):
    links = utils.load_pickle(model_path + dataset_name + '_links.pkl')
    with open(model_path + dataset_name + "_links", "w") as f:
        for did in links:
            f.write(str(did) + "\t" + " ".join(
                map(lambda m1, m2: str(m1) + "," + str(m2), links[did])) + "\n")


def main(model_path, dataset_name):
    docs = utils.load_pickle(model_path + dataset_name + '_processed_docs.pkl')

    for doc_data in utils.load_json_lines(directories.RAW + dataset_name):
        sentences = doc_data["sentences"]
        mid_to_mention = {int(m["mention_id"]): m for m in doc_data["mentions"].values()}
        mid_to_position = {mid: int(m["mention_num"]) for mid, m in mid_to_mention.iteritems()}

        doc = docs[doc_data["document_features"]["doc_id"]]
        clusters = [c for c in doc.clusters if len(c) > 1]

        cluster_to_endpoints = {}
        for c in clusters:
            positions = [mid_to_position[mid] for mid in c]
            cluster_to_endpoints[c] = (min(positions), max(positions))
        sorted_clusters = sorted(clusters, key=lambda c: cluster_to_endpoints[c])

        color_last_usage = {i: -1 for i in range(len(COLORS))}
        active_clusters = []
        cluster_to_color = {}
        for c in sorted_clusters:
            start, end = cluster_to_endpoints[c]
            for a in list(active_clusters):
                if cluster_to_endpoints[a][1] < start:
                    active_clusters.remove(a)

            used_colors = [cluster_to_color[a] for a in active_clusters]
            sorted_colors = sorted((u, i) for i, u in color_last_usage.iteritems())
            next_color = None
            for u, i in sorted_colors:
                if i not in used_colors:
                    next_color = i
                    break
            if next_color is None:
                next_color = sorted_colors[0][1]

            color_last_usage[next_color] = start
            cluster_to_color[c] = next_color
            active_clusters.append(c)

        annotations = defaultdict(lambda: defaultdict(list))
        for i, c in enumerate(sorted_clusters):
            color = COLORS[cluster_to_color[c]]
            for m in c:
                mention = mid_to_mention[m]
                start, end = mention["start_index"], mention["end_index"] - 1
                annotations[mention["sent_num"]][start].append(
                    (color + "[" + ENDC, 1 + end))
                annotations[mention["sent_num"]][end].append(
                    (color + "]" + subscript(i) + ENDC, -1 - start))

        for i, s in enumerate(sentences):
            for j, sentence_annotations in annotations[i].iteritems():
                sentence_annotations = sorted(sentence_annotations, key=itemgetter(1))
                for (annotation, priority) in sentence_annotations:
                    if priority > 0:
                        s[j] = annotation + s[j]
                    else:
                        s[j] = s[j] + annotation
            print(" ".join(s))

        print()
        print(80 * "=")
        print()


if __name__ == '__main__':
    main(directories.MODELS + 'reward_rescaling/', 'test')
