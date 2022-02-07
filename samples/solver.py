import os
import sys
import numpy as np

# размеры результирующего изображения
W = 1200
H = 900
CHANNEL_NUM = 3  # мы работаем с изображениями rgb
MAX_VALUE = 255  # максимальное значение пикселя, требуемое заголовком ppm


def read_image(path):
    # вторая строка заголовка содержит размеры изображения
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # пропустить 3 строки, зарезервированные для заголовка и чтения изображения
    image = np.loadtxt(path, skiprows=3, dtype=np.uint8).reshape((h, w, CHANNEL_NUM))
    # print("Изображение: \n", image)
    return image


def write_image(path, img):
    h, w = img.shape[:2]
    # формат ppm требует заголовка в специальном формате
    header = f'P3\n{w} {h}\n{MAX_VALUE}\n'
    with open(path, 'w') as f:
        f.write(header)
        for r, g, b in img.reshape((-1, CHANNEL_NUM)):
            # print(r,g,b)
            f.write(f'{r} {g} {b} ')


def solve_puzzle(tiles_folder):
    # создать заполнитель для результирующего изображения
    # прочитать все плитки в списке
    tiles = [read_image(os.path.join(tiles_folder, t)) for t in sorted(os.listdir(tiles_folder))]
    result_img = np.zeros((H, W, CHANNEL_NUM), dtype=np.uint8)
    # отсканируйте размеры всех плиток и найдите минимальную высоту и ширину
    dims = np.array([t.shape[:2] for t in tiles])
    h, w = np.min(dims, axis=0)
    # вычислить сетку, которая будет покрывать изображение
    # расстояние между рядами сетки = min h
    # расстояние между столбцами сетки = min w
    x_nodes = np.arange(0, W, w)
    y_nodes = np.arange(0, H, h)
    xx, yy = np.meshgrid(x_nodes, y_nodes)
    nodes = np.vstack((xx.flatten(), yy.flatten())).T
    # заполнить сетку плиткой
    for (x, y), tile in zip(nodes, tiles):
        result_img[y: y + h, x: x + w] = tile[:h, :w]

    output_path = "image.ppm"
    write_image(output_path, result_img)


if __name__ == "__main__":
    directory = "/Users/lebedevila/PycharmProjects/machine_learning_testovoe/data/0000_0000_0000/tiles"
    # directory = sys.argv[1]
    solve_puzzle(directory)
