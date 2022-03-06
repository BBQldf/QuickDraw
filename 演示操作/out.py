import pygame
from pygame.locals import *
import math
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class Brush:
    def __init__(self, screen):
        self.screen = screen
        self.color = (0, 0, 0)
        self.size = 16
        self.drawing = False
        self.last_pos = None
        self.style = True
        self.brush = pygame.image.load("images/brush.png").convert_alpha()
        self.brush_now = self.brush.subsurface((0, 0), (1, 1))

    def start_draw(self, pos):
        self.drawing = True
        self.last_pos = pos

    def end_draw(self):
        self.drawing = False

    def set_brush_style(self, style):
        print("* set brush style to", style)
        self.style = style

    def get_brush_style(self):
        return self.style

    def get_current_brush(self):
        return self.brush_now

    def set_size(self, size):
        if size < 1:
            size = 1
        elif size > 32:
            size = 32
        print("* set brush size to", size)
        self.size = size
        self.brush_now = self.brush.subsurface((0, 0), (size * 2, size * 2))

    def get_size(self):
        return self.size

    def set_color(self, color):
        self.color = color
        for i in range(self.brush.get_width()):
            for j in range(self.brush.get_height()):
                self.brush.set_at((i, j),
                                  color + (self.brush.get_at((i, j)).a,))

    def get_color(self):
        return self.color

    def draw(self, pos):
        if self.drawing:
            for p in self._get_points(pos):
                if self.style:
                    self.screen.blit(self.brush_now, p)
                else:
                    pygame.draw.circle(self.screen, self.color, p, self.size)
            self.last_pos = pos

    def _get_points(self, pos):
        points = [(self.last_pos[0], self.last_pos[1])]
        len_x = pos[0] - self.last_pos[0]
        len_y = pos[1] - self.last_pos[1]
        length = math.sqrt(len_x ** 2 + len_y ** 2)
        step_x = len_x / length
        step_y = len_y / length
        for i in range(int(length)):
            points.append((points[-1][0] + step_x, points[-1][1] + step_y))
        points = map(lambda x: (int(0.5 + x[0]), int(0.5 + x[1])), points)
        return list(set(points))


class Menu:
    def __init__(self, screen):
        self.usemodel = Usemodel()

        self.screen = screen
        self.brush = None
        self.colors = [
            (0xff, 0x00, 0xff), (0x80, 0x00, 0x80),
            (0x00, 0x00, 0xff), (0x00, 0x00, 0x80),
            (0x00, 0xff, 0xff), (0x00, 0x80, 0x80),
            (0x00, 0xff, 0x00), (0x00, 0x80, 0x00),
            (0xff, 0xff, 0x00), (0x80, 0x80, 0x00),
            (0xff, 0x00, 0x00), (0x80, 0x00, 0x00),
            (0xc0, 0xc0, 0xc0), (0xff, 0xff, 0xff),
            (0x00, 0x00, 0x00), (0x80, 0x80, 0x80),
        ]
        self.colors_rect = []
        for (i, rgb) in enumerate(self.colors):
            rect = pygame.Rect(10 + i % 2 * 32, 254 + i / 2 * 32, 32, 32)
            self.colors_rect.append(rect)
        self.pens = [
            pygame.image.load("images/pen1.png").convert_alpha(),
            pygame.image.load("images/pen2.png").convert_alpha(),
        ]
        self.pens_rect = []
        for (i, img) in enumerate(self.pens):
            rect = pygame.Rect(10, 10 + i * 64, 64, 64)
            self.pens_rect.append(rect)
        self.sizes = [
            pygame.image.load("images/big.png").convert_alpha(),
            pygame.image.load("images/small.png").convert_alpha()
        ]
        self.sizes_rect = []
        for (i, img) in enumerate(self.sizes):
            rect = pygame.Rect(10 + i * 32, 138, 32, 32)
            self.sizes_rect.append(rect)
        self.savedraw = pygame.image.load("images/Save.png").convert_alpha()
        self.savedraw_rect = pygame.Rect(10, 10 + 530, 64, 64)

    def set_brush(self, brush):
        self.brush = brush

    def draw(self):
        self.screen.blit(self.savedraw, self.savedraw_rect.topleft)
        for (i, img) in enumerate(self.pens):
            self.screen.blit(img, self.pens_rect[i].topleft)
        for (i, img) in enumerate(self.sizes):
            self.screen.blit(img, self.sizes_rect[i].topleft)
        self.screen.fill((255, 255, 255), (10, 180, 64, 64))
        pygame.draw.rect(self.screen, (0, 0, 0), (10, 180, 64, 64), 1)
        size = self.brush.get_size()
        x = 10 + 32
        y = 180 + 32
        if self.brush.get_brush_style():
            x = x - size
            y = y - size
            self.screen.blit(self.brush.get_current_brush(), (x, y))
        else:
            pygame.draw.circle(self.screen,
                               self.brush.get_color(), (x, y), size)
        for (i, rgb) in enumerate(self.colors):
            pygame.draw.rect(self.screen, rgb, self.colors_rect[i])

    def save_png(self):
        pygame.image.save(self.screen, "user_draw.png")
        word = self.usemodel.predictpng()
        self.showtext(word)

    def showtext(self, thetext):
        pygame.init()
        text = pygame.font.SysFont("timesnewroman", 30)
        text = text.render(thetext, 1, (0, 0, 0))
        self.screen.blit(text, (10, 560))

    def click_button(self, pos):
        for (i, rect) in enumerate(self.pens_rect):
            if rect.collidepoint(pos):
                self.brush.set_brush_style(bool(i))
                return True
        for (i, rect) in enumerate(self.sizes_rect):
            if rect.collidepoint(pos):
                if i:
                    self.brush.set_size(self.brush.get_size() - 1)
                else:
                    self.brush.set_size(self.brush.get_size() + 1)
                return True
        for (i, rect) in enumerate(self.colors_rect):
            if rect.collidepoint(pos):
                self.brush.set_color(self.colors[i])
                return True
        if self.savedraw_rect.collidepoint(pos):
            self.save_png()
            return True

        return False


class Painter:
    def __init__(self):
        self.screen = pygame.display.set_mode((674, 600))
        pygame.display.set_caption("Painter")
        self.clock = pygame.time.Clock()
        self.brush = Brush(self.screen)
        self.menu = Menu(self.screen)
        self.menu.set_brush(self.brush)

    def run(self):
        self.screen.fill((255, 255, 255))
        while True:
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.screen.fill((255, 255, 255))
                elif event.type == MOUSEBUTTONDOWN:
                    if event.pos[0] <= 74 and self.menu.click_button(event.pos):
                        pass
                    else:
                        self.brush.start_draw(event.pos)
                elif event.type == MOUSEMOTION:
                    self.brush.draw(event.pos)
                elif event.type == MOUSEBUTTONUP:
                    self.brush.end_draw()
            self.menu.draw()
            pygame.display.update()


class Usemodel:
    def __init__(self):
        self.model = tf.keras.models.load_model("mymodel.h5")
        self.class_names = ['airplane', 'alarm_clock', 'anvil', 'apple', 'axe',
                            'baseball', 'baseball_bat', 'basketball', 'beard', 'bed',
                            'bench', 'bicycle', 'bird', 'book', 'bread',
                            'bridge', 'broom', 'butterfly', 'camera', 'candle',
                            'car', 'cat', 'ceiling_fan', 'cell_phone', 'chair',
                            'circle', 'clock', 'cloud', 'coffee_cup', 'cookie',
                            'cup', 'diving_board', 'donut', 'door', 'drums',
                            'dumbbell', 'envelope', 'eye', 'eyeglasses', 'face',
                            'fan', 'flower', 'frying_pan', 'grapes', 'hammer',
                            'hat', 'headphones', 'helmet', 'hot_dog', 'ice_cream',
                            'key', 'knife', 'ladder', 'laptop', 'lightning',
                            'light_bulb', 'line', 'lollipop', 'microphone', 'moon',
                            'mountain', 'moustache', 'mushroom', 'pants', 'paper_clip',
                            'pencil', 'pillow', 'pizza', 'power_outlet', 'radio',
                            'rainbow', 'rifle', 'saw', 'scissors', 'screwdriver',
                            'shorts', 'shovel', 'smiley_face', 'snake', 'sock',
                            'spider', 'spoon', 'square', 'star', 'stop_sign',
                            'suitcase', 'sun', 'sword', 'syringe', 't-shirt',
                            'table', 'tennis_racquet', 'tent', 'tooth', 'traffic_light',
                            'tree', 'triangle', 'umbrella', 'wheel', 'wristwatch']

    def predictpng(self):
        im = Image.open("user_draw.png")
        im = im.crop((74, 0, 674, 600))  # (left, upper, right, lower)
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = im.convert('1')
        im.save('123.png')
        imbit = []
        for y in range(28):
            for x in range(28):
                imbit.append(255 - im.getpixel((x, y)))
        imbit = np.array(imbit)
        imbit = imbit.reshape((28, 28, 1)).astype("float32")
        imbit /= 255.0


        img = imbit
        plt.imshow(img.squeeze())
        pred = self.model.predict(np.expand_dims(img, axis=0))[0]
        ind = (-pred).argsort()[:5]
        latex = [self.class_names[x] for x in ind]
        print(latex)
        return latex[0]


def main():
    app = Painter()
    app.run()


if __name__ == '__main__':
    main()