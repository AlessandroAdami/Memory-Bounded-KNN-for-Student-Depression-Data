from manim import *
import random
from collections import Counter

class Knn(Scene):
    def construct(self):
        n = 50 
        k = 5
        dots = []
        lines = []

        for _ in range(n):
            x = random.uniform(-6, 6)
            y = random.uniform(-3.4, 3.4)
            if -0.2 < x < 0.2 and -0.2 < y < 0.2:
                continue
            if x > 0:
                if random.uniform(0,1) > 0.8:
                    c = BLUE
                else:
                    c = RED
            else:
                if random.uniform(0,1) > 0.8:
                    c = RED
                else:
                    c = BLUE
            dot = Dot(point=[x, y, 0], color=c)
            dots.append(dot)

        for dot in dots:
            self.play(FadeIn(dot), run_time=0.1)

        self.wait(3)

        center_p = Dot(point=ORIGIN, color=WHITE,radius=0.15)
        self.play(FadeIn(center_p))

        for dot in dots:
            line = Line(dot.get_center(), center_p.get_center(), color=WHITE)
            self.bring_to_back(line)
            self.play(Create(line), run_time=0.1)
            lines.append((line, dot))

        self.wait(3)

        lines.sort(key=lambda pair: np.linalg.norm(pair[1].get_center() - center_p.get_center()))

        to_keep = lines[:k]
        to_remove = lines[k:]

        self.play(*[FadeOut(line) for line, _ in to_remove], run_time=2)

        self.wait(2)

        colors = [dot.get_color() for _, dot in to_keep]


        color_counts = Counter(colors)
        most_common_color, _ = color_counts.most_common(1)[0]

        for _, dot in to_keep:
            if dot.get_color() == most_common_color:
                self.play(Indicate(dot),color=WHITE)

        self.play(center_p.animate.set_color(most_common_color), run_time=0.5)

        self.play(*[FadeOut(line) for line, _ in to_keep])


class KnnMemoryBounded(Scene):
    def construct(self):
        n = 50 
        k = 5
        dots = []
        lines = []
        bad_dots = []

        for _ in range(n):
            x = random.uniform(-6, 6)
            y = random.uniform(-3.4, 3.4)
            is_bad_dot = False
            if -0.2 < x < 0.2 and -0.2 < y < 0.2:
                continue
            if x > 0:
                if random.uniform(0,1) > 0.8:
                    c = BLUE
                    is_bad_dot = True
                else:
                    c = RED
            else:
                if random.uniform(0,1) > 0.8:
                    c = RED
                    is_bad_dot = True
                else:
                    c = BLUE
            dot = Dot(point=[x, y, 0], color=c)
            if random.uniform(0,1) > 0.4: is_bad_dot = True
            dots.append(dot)
            if is_bad_dot:
                bad_dots.append(dot)

        self.play(*[FadeIn(dot) for dot in dots])

        self.wait(3)

        for dot in bad_dots:
            self.play(FadeOut(dot),run_time=0.2)
            dots.remove(dot)

        self.wait(1)

        center_p = Dot(point=ORIGIN, color=WHITE,radius=0.15)
        self.play(FadeIn(center_p))

        self.wait(1)

        for dot in dots:
            line = Line(dot.get_center(), center_p.get_center(), color=WHITE)
            self.bring_to_back(line)
            self.play(Create(line), run_time=0.05)
            lines.append((line, dot))

        self.wait(1.5)

        lines.sort(key=lambda pair: np.linalg.norm(pair[1].get_center() - center_p.get_center()))

        to_keep = lines[:k]
        to_remove = lines[k:]

        self.play(*[FadeOut(line) for line, _ in to_remove], run_time=2)

        self.wait(3)

        colors = [dot.get_color() for _, dot in to_keep]

        color_counts = Counter(colors)
        most_common_color, _ = color_counts.most_common(1)[0]

        for _, dot in to_keep:
            if dot.get_color() == most_common_color:
                self.play(Indicate(dot),color=WHITE)

        self.play(center_p.animate.set_color(most_common_color), run_time=0.5)

        self.play(*[FadeOut(line) for line, _ in to_keep])