# Exercise 4
def ex4():
    from matplotlib import pyplot as plt
    import pylab as plb

    plb.rcParams["font.size"] = 16

    def mk_groups(data):
        try:
            newdata = data.items()
        except:
            return

        thisgroup = []
        groups = []
        for key, value in newdata:
            newgroups = mk_groups(value)
            if newgroups is None:
                thisgroup.append((key, value))
            else:
                thisgroup.append((key, len(newgroups[-1])))
                if groups:
                    groups = [g + n for n, g in zip(newgroups, groups)]
                else:
                    groups = newgroups
        return [thisgroup] + groups

    def add_line(ax, xpos, ypos):
        line = plt.Line2D(
            [xpos, xpos], [ypos + 0.1, ypos], transform=ax.transAxes, color="black"
        )
        line.set_clip_on(False)
        ax.add_line(line)

    def label_group_bar(ax, data):
        groups = mk_groups(data)
        xy = groups.pop()
        x, y = zip(*xy)
        ly = len(y)
        xticks = range(1, ly + 1)

        ax.bar(xticks, y, align="center")
        ax.set_xticks(xticks)
        ax.set_xticklabels(x)
        ax.set_xlim(0.5, ly + 0.5)
        ax.yaxis.grid(True)

        scale = 1.0 / ly
        for pos in range(ly + 1):
            add_line(ax, pos * scale, -0.1)
        ypos = -0.2  # Adjust this to shift the bottom labels
        while groups:
            group = groups.pop()
            pos = 0
            for label, rpos in group:
                lxpos = (pos + 0.5 * rpos) * scale
                ax.text(
                    lxpos,
                    ypos,
                    label,
                    ha="center",
                    transform=ax.transAxes,
                    rotation=70,
                    color="red",
                )
                add_line(ax, pos * scale, ypos)
                pos += rpos
            add_line(ax, pos * scale, ypos)
            ypos -= 0.1

    dataDict = {
        # Set attributes and level part worths for the attribute.
        "Car Safety": {"High": 4.666666667, "Good": 5, "Average": 5.333333333},
        "Fuel": {"18km/l": 5.333333333, "15km/l": 5, "13km/l": 4.666666667},
        "Accessories": {"SiriusXm": 3, "BackupCam": 4, "Heated Seats": 8},
    }
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1)
    label_group_bar(ax, dataDict)
    plt.title("Part Worths")
    plt.show()


# Exercise 5
def ex5():
    import matplotlib.pyplot as plt

    carAttributes = ["Saftey", "Fuel Economy", "Accessories"]
    importanceLevels = [10.52631579, 10.52631579, 78.94736842]
    plt.bar(carAttributes, importanceLevels)
    plt.xticks(rotation=75)
    plt.title("Car Attribute Importance")
    plt.show()


def main():
    # ex4()
    ex5()


if __name__ == "__main__":
    main()
