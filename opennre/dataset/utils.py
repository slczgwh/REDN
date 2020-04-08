def is_normal(epl):
    entities = set()
    for e in epl:
        entities.add(e[0])
        entities.add(e[1])
    return len(entities) == (len(epl) * 2)


def is_multi_label(epl):
    if is_normal(epl):
        return False
    entities_pair = []
    for i, e in enumerate(epl):
        entities_pair.append(tuple([e[0], e[1]]))
    return len(entities_pair) != len(set(entities_pair))


def is_over_lapping(epl):
    if is_normal(epl):
        return False

    entities_pair = []
    for i, e in enumerate(epl):
        entities_pair.append(tuple([e[0], e[1]]))

    entities_pair = set(entities_pair)
    entities = []
    for pair in entities_pair:
        entities.extend(pair)
    entities = set(entities)
    return len(entities) != (2 * len(entities_pair))
