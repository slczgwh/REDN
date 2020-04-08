import json
import os
import os.path as osp
from shutil import copyfile
import nltk

root_path = "/mnt/nas1/users/tianye/projects/copy_re/data/webnlg/entity_end_position"
orig_path = "/mnt/nas1/pare/benchmark/webnlg_orig"
output_path = "/mnt/nas1/pare/benchmark/webnlg"

words2id = json.load(open(osp.join(root_path, "words2id.json"), "r", encoding="UTF-8"))
rel2id = json.load(open(osp.join(root_path, "relations2id.json"), "r", encoding="UTF-8"))
id2words = {value: key for key, value in words2id.items()}
id2rel = {value: key for key, value in rel2id.items()}


def parse(input_f, output_f):
    f = json.load(open(input_f, "r", encoding="UTF-8"))
    outf = open(output_f, "w", encoding="UTF-8")

    for sen_id, trp in zip(f[0], f[1]):
        tokens = []
        for _id in sen_id:
            tokens.append(id2words[_id])

        for i in range(len(trp) // 3):
            res = {
                "token": tokens,
                "relation": id2rel[trp[i * 3 + 2]],
                "h": {"pos": [trp[i * 3 + 0], trp[i * 3 + 0] + 1]},
                "t": {"pos": [trp[i * 3 + 1], trp[i * 3 + 1] + 1]}
            }
            outf.write(json.dumps(res))
            outf.write("\n")
    outf.flush()
    outf.close()


def parse_original(input_f, output_f, build_rel2id=False):
    rel2id = {}
    rel_idx = 0
    outf = open(output_f, "w", encoding="UTF-8")
    line_count = 0
    error_count = 0
    orig_data = json.load(open(osp.join(input_f), "r", encoding="UTF-8"))
    for idx, entry in enumerate(orig_data["entries"]):
        e = entry[str(idx + 1)]
        triple_list = []
        for triple in e["modifiedtripleset"]:
            head = nltk.word_tokenize(triple["subject"].replace("_", " "))
            rel = triple["property"]
            tail = nltk.word_tokenize(triple["object"].replace("_", " "))
            if rel not in rel2id:
                rel2id[rel] = rel_idx
                rel_idx += 1

            triple_list.append([head, tail, rel])
        for lexs in e["lexicalisations"]:
            sentense = lexs["lex"]
            # tokens = sentense.split(" ")
            tokens = nltk.word_tokenize(sentense)
            for triple in triple_list:
                triple_pos = [[-1, -1], [-1, -1]]
                lower_tokens = [t.lower() for t in tokens]
                for j in range(2):
                    index = 0
                    while True:
                        try:
                            index = lower_tokens.index(triple[j][0].lower(), index)
                        except ValueError:
                            break
                        for i in range(len(triple[j])):
                            try:
                                if lower_tokens[index + i] != triple[j][i].lower():
                                    break
                            except IndexError:
                                break
                        else:
                            triple_pos[j] = [index, index + len(triple[j])]
                        break
                if min(min(triple_pos)) > -1:
                    _d = {"token": tokens,
                          "h": {"name": " ".join(triple[0]), "pos": [triple_pos[0][0], triple_pos[0][1]]},
                          "t": {"name": " ".join(triple[1]), "pos": [triple_pos[1][0], triple_pos[1][1]]},
                          "relation": triple[2]
                          }
                    outf.write(json.dumps(_d))
                    outf.write("\n")
                    line_count += 1
                else:
                    error_count += 1
                    print("\r %d/%d" % (error_count, line_count), end="")
    print("write line %d" % line_count)
    outf.flush()
    outf.close()

    if build_rel2id:
        rel_f = osp.split(output_f)
        with open(osp.join(rel_f[0], "rel2id.json"), "w", encoding="UTF-8") as f:
            f.write(json.dumps(rel2id))


# for fname, ofname in zip(["train.json", "valid.json", "dev.json"], ["train.json", "dev.json", "test.json"]):
#     parse(osp.join(root_path, fname), osp.join(output_path, ofname))
#     copyfile(osp.join(root_path, "relations2id.json"), osp.join(output_path, "rel2id.json"))

for i, (fname, ofname) in enumerate(
        zip(["webnlg_release_v2_train.json", "webnlg_release_v2_dev.json", "webnlg_release_v2_test.json"],
            ["train.json", "dev.json", "test.json"])):
    parse_original(osp.join(orig_path, fname), osp.join(orig_path, ofname), build_rel2id=i == 0)
