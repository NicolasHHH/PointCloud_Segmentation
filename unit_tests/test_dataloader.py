from data_loaders.ShapeNet import PartNormalDataset
from data_loaders.constants import shapenetpart_cat2id, seg_classes


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
