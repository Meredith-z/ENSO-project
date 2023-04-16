PARA = dict(
    data=dict(
        MAX=31.194440841674805,
        MIN=17.907947540283203
    ),
    train=dict(
        epochs=100,
        batch_size=32,
        lr=1e-4,
        momentum=0.9,
        wd=5e-4,
        num_workers=0,
        divice_ids=[1],
        gpu_id=0,
        num_classes=10,
    ),
    test=dict(
        batch_size=16
    ),
    enso_paths=dict(
        validation_rate=0.05,

        root='../../DATASET/cifar10/',

        original_trainset_path='../../../DATASET/cifar10/cifar-10-python/',  # train_batch_path
        original_testset_path='../../../DATASET/cifar10/cifar-10-python/',

        after_trainset_path='../../../DATASET/cifar10/trainset/',
        after_testset_path='../../../DATASET/cifar10/testset/',
        after_validset_path='../../../DATASET/cifar10/validset/',

        train_CMIP_input='../DataProcess/train_X_CMIP.pt',
        train_CMIP_target='../DataProcess/train_Y_CMIP.pt',
        valid_CMIP_input='../DataProcess/test_X_CMIP.pt',
        valid_CMIP_target='../DataProcess/test_Y_CMIP.pt',

    ),
    utils_paths=dict(
        checkpoint_path='./cache/checkpoint/',
        log_path='./cache/log/',
        visual_path='./cache/visual/',
        params_path='./cache/params/',
        result_path='./cache/result/',
    ),

    predrnn=dict(
        input_length=12,
        total_length=24,
        img_height=20,
        img_width=50,
        img_channel=1,
        device='cuda',

        # model
        num_layers=4,
        num_hidden=[64, 64, 64, 64],
        filter_size=5,
        stride=1,
        patch_size=1,
        layer_norm=1,
        decouple_beta=0.1,

        # reverse scheduled sampling
        reverse_scheduled_sampling=0,
        r_sampling_step_1=25000,
        r_sampling_step_2=50000,
        r_exp_alpha=5000,

        # scheduled sampling
        scheduled_sampling=1,
        sampling_stop_iter=50000,
        sampling_start_value=1.0,
        sampling_changing_rate=0.00002,

        # optimization
        reverse_input=1,
        batch_size=16,
        max_iterations=80000,
        display_interval=100,
        test_interval=5000,
        snapshot_interval=5000,
        num_save_samples=10,
        n_gpu=1,

        # visualization of memory decoupling
        visual=0,
        visual_path='./decoupling_visual'
    )
)

encoder_config = [('conv', 'leaky', 1, 16, 3, 1, 2),
             ('convlstm', '', 16, 16, 3, 1, 1),
             ('conv', 'leaky', 16, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1)]
import os
for path in PARA.utils_paths.values():
    if not os.path.exists(path):
        os.makedirs(path)