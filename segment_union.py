import bisect


class SegmentsUnion:
    def __init__(self):
        self.segments_axis = []

    def add(self, new_segment):
        new_a, new_b = new_segment
        bisect.insort(self.segments_axis, (new_a, False), key=lambda val_flag: val_flag)
        bisect.insort(self.segments_axis, (new_b, True), key=lambda val_flag: val_flag)

    def get_union(self):
        flag_counter = 0
        segment_started = None

        union = []
        for value, flag in self.segments_axis:
            if segment_started is not None:
                if not flag:
                    flag_counter += 1
                else:
                    flag_counter -= 1
                if flag_counter == 0:
                    union.append([segment_started, value])
                    segment_started = None
            else:
                if not flag:
                    segment_started = value
                    flag_counter += 1
                else:
                    print('WTF?')
                    continue

        return union
