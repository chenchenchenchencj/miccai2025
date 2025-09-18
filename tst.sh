for fold in {0..4}
do
    echo "nnUNetv2_train 1 3d_lowres $fold"
    nnUNetv2_train 1 3d_lowres $fold
done
