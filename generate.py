import os
import random
import sys

from typing import Tuple


def get_matrix(fill='.'):
    values = list()
    indices = list()
    for i in range(4):
        level = list()
        for j in range(4):
            indices.append((i, j))
            level.append(fill)
        values.append(level)
    return values, indices


def fill_matrix(matrix, indices, fill='.'):
    for (i, j) in indices:
        matrix[i][j] = fill
    return matrix


def matrix2str(matrix, sep=' '):
    s = ""
    for level in matrix:
        s += sep.join(level) + '\n'
    return s


def _generate_one() -> Tuple[str, Tuple[int, int]]:
    matrix, indices = get_matrix()

    # Generate all the possible "win" lines where Xs will be generated.
    h_axis = [list(filter(lambda ij: ij[0] == row_idx, indices)) for row_idx in range(4)]
    v_axis = [list(filter(lambda ij: ij[1] == col_idx, indices)) for col_idx in range(4)]
    m_diag = list(filter(lambda ij: ij[0] == ij[1], indices))
    i_diag = list(filter(lambda ij: ij[1] == 3 - ij[0], indices))

    # Choice one of them.
    answer_axis = random.choice([m_diag, i_diag, *h_axis, *v_axis])
    outer_indices = list(set(indices).difference(answer_axis))

    # Save one idx as an answer.
    answer_idx = random.choice(answer_axis)

    # Draw Xs onto the matrix.
    answer_axis.remove(answer_idx)
    fill_matrix(matrix, answer_axis, fill='X')

    # Fill the outer indices with Os, Xs, and dots.
    k_o = random.choice(range(3, 8))

    # 4 * 4 = 16, 4 - used by Xs already, k_o will be used by Os.
    k_x = max(3, min(16 - 4 - k_o, 7))

    o_indices = random.choices(outer_indices, k=k_o)
    x_indices = random.choices(list(set(outer_indices).difference(o_indices)), k=k_x)

    fill_matrix(matrix, o_indices, fill='O')
    fill_matrix(matrix, x_indices, fill='X')

    return matrix, answer_idx



def generate(n=1, save_path: str = './'):
    random.seed(666)
    examples = list()
    answers = list()
    
    for i in range(n):
        example, answer = _generate_one()
        
        # Regenerate if example already exists.
        while example in examples:
           example, answer = _generate_one() 
           print('Regenerating example %d ...' % i)
        
        examples.append(matrix2str(example, sep=''))
        answers.append(' '.join(map(str, answer)))
    
    with open(os.path.join(save_path, 'trainData.txt'), 'w') as file:
        file.writelines(examples)

    with open(os.path.join(save_path, 'trainLabels.txt'), 'w') as file:
        file.writelines('\n'.join(answers))


if __name__ == '__main__':
    generate(n=int(sys.argv[1]))
