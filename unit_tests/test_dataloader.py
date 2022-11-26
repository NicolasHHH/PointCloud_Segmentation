from data_loaders.ShapeNet import PartNormalDataset

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

shapenetpart_cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}

def main():
    # test 1 .__len__()
    print("test 1 .__len__()\n")
    dataloader = PartNormalDataset(seg_classes, root="../data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
                                   split="trainval")
    print("trainval : ", len(dataloader))
    dataloader = PartNormalDataset(seg_classes, root="../data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
                                   split="val")
    print("val : ", len(dataloader))
    dataloader = PartNormalDataset(seg_classes, root="../data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
                                   split="test")
    print("test : ", len(dataloader))
    dataloader = PartNormalDataset(seg_classes, root="../data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
                                   split="train")
    print("train : ", len(dataloader))

    # test 2 .__getItem__()
    print("\ntest 2 .__getItem__()\n")
    point_set, cls, seg = dataloader[1]
    print(point_set.shape, point_set.dtype)
    print(cls.shape, cls.dtype)
    print(seg.shape, seg.dtype)

    # test 3 use_normals
    print("\ntest 3 .__getItem__()\n")
    dataloader = PartNormalDataset(seg_classes, root="../data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
                                   split="train", use_normals=True)
    point_set, cls, seg = dataloader[2]
    # (n_points * 6)
    print(point_set.shape, point_set.dtype)


if __name__ == '__main__':
    main()
