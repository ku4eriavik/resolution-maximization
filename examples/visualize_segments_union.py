from segment_union import SegmentsUnion


if __name__ == '__main__':
    seg_un = SegmentsUnion()

    segments = [
        (4, 8),
        (9, 12),
        (15, 17),
        (2, 3),
        (10, 11),
        (13, 14),
        (18, 20),
        (21, 23),
        (10, 19),
        (8.5, 24),
        (0, 6),
        (16, float('inf')),
        (float('-inf'), -2)
    ]
    for seg in segments:
        seg_un.add(seg)

    union = seg_un.get_union()
    print(union)
