args = None

DomainNetOpt = {
    'name': 'domainnet',
    'batch_size': 32,
    'learning_rate': 0.00005,
    'file_path': './dataset/Domainnet',
    'classes': ['The_Eiffel_Tower', 'boomerang', 'coffee_cup', 'fireplace', 'house', 'moustache',
                'postcard', 'snorkel', 'tiger',
                'The_Great_Wall_of_China', 'bottlecap', 'compass', 'firetruck', 'house_plant',
                'mouth', 'potato', 'snowflake', 'toaster',
                'The_Mona_Lisa', 'bowtie', 'computer', 'fish', 'hurricane', 'mug', 'power_outlet',
                'snowman', 'toe',
                'aircraft_carrier', 'bracelet', 'cookie', 'flamingo', 'ice_cream', 'mushroom',
                'purse', 'soccer_ball', 'toilet'
        , 'airplane', 'brain', 'cooler', 'flashlight', 'jacket', 'nail', 'rabbit', 'sock', 'tooth'
        , 'alarm_clock', 'bread', 'couch', 'flip_flops', 'jail', 'necklace', 'raccoon', 'speedboat', 'toothbrush'
        , 'ambulance', 'bridge', 'cow', 'floor_lamp', 'kangaroo', 'nose', 'radio', 'spider', 'toothpaste'
        , 'angel', 'broccoli', 'crab', 'flower', 'key', 'ocean', 'rain', 'spoon', 'tornado'
        , 'animal_migration', 'broom', 'crayon', 'flying_saucer', 'keyboard', 'octagon', 'rainbow', 'spreadsheet',
                'tractor'
        , 'ant', 'bucket', 'crocodile', 'foot', 'knee', 'octopus', 'rake', 'square', 'traffic_light'
        , 'anvil', 'bulldozer', 'crown', 'fork', 'knife', 'onion', 'remote_control', 'squiggle', 'train'
        , 'apple', 'bus', 'cruise_ship', 'frog', 'ladder', 'oven', 'rhinoceros', 'squirrel', 'tree'
        , 'arm', 'bush', 'cup', 'frying_pan', 'lantern', 'owl', 'rifle', 'stairs', 'triangle'
        , 'asparagus', 'butterfly', 'diamond', 'garden', 'laptop', 'paint_can', 'river', 'star', 'trombone'
        , 'axe', 'cactus', 'dishwasher', 'garden_hose', 'leaf', 'paintbrush', 'roller_coaster', 'steak', 'truck'
        , 'backpack', 'cake', 'diving_board', 'giraffe', 'leg', 'palm_tree', 'rollerskates', 'stereo', 'trumpet'
        , 'banana', 'calculator', 'dog', 'goatee', 'light_bulb', 'panda', 'sailboat', 'stethoscope', 'umbrella'
        , 'bandage', 'calendar', 'dolphin', 'golf_club', 'lighter', 'pants', 'sandwich', 'stitches', 'underwear'
        , 'barn', 'camel', 'donut', 'grapes', 'lighthouse', 'paper_clip', 'saw', 'stop_sign', 'van'
        , 'baseball', 'camera', 'door', 'grass', 'lightning', 'parachute', 'saxophone', 'stove', 'vase'
        , 'baseball_bat', 'camouflage', 'dragon', 'guitar', 'line', 'parrot', 'school_bus', 'strawberry', 'violin'
        , 'basket', 'campfire', 'dresser', 'hamburger', 'lion', 'passport', 'scissors', 'streetlight', 'washing_machine'
        , 'basketball', 'candle', 'drill', 'hammer', 'lipstick', 'peanut', 'scorpion', 'string_bean', 'watermelon'
        , 'bat', 'cannon', 'drums', 'hand', 'lobster', 'pear', 'screwdriver', 'submarine', 'waterslide'
        , 'bathtub', 'canoe', 'duck', 'harp', 'lollipop', 'peas', 'sea_turtle', 'suitcase', 'whale'
        , 'beach', 'car', 'dumbbell', 'hat', 'mailbox', 'pencil', 'see_saw', 'sun', 'wheel'
        , 'bear', 'carrot', 'ear', 'headphones', 'map', 'penguin', 'shark', 'swan', 'windmill'
        , 'beard', 'castle', 'elbow', 'hedgehog', 'marker', 'piano', 'sheep', 'sweater', 'wine_bottle'
        , 'bed', 'cat', 'elephant', 'helicopter', 'matches', 'pickup_truck', 'shoe', 'swing_set', 'wine_glass'
        , 'bee', 'ceiling_fan', 'envelope', 'helmet', 'megaphone', 'picture_frame', 'shorts', 'sword', 'wristwatch'
        , 'belt', 'cell_phone', 'eraser', 'hexagon', 'mermaid', 'pig', 'shovel', 'syringe', 'yoga'
        , 'bench', 'cello', 'eye', 'hockey_puck', 'microphone', 'pillow', 'sink', 't-shirt', 'zebra'
        , 'bicycle', 'chair', 'eyeglasses', 'hockey_stick', 'microwave', 'pineapple', 'skateboard', 'table', 'zigzag'
        , 'binoculars', 'chandelier', 'face', 'horse', 'monkey', 'pizza', 'skull', 'teapot'
        , 'bird', 'church', 'fan', 'hospital', 'moon', 'pliers', 'skyscraper', 'teddy-bear'
        , 'birthday_cake', 'circle', 'feather', 'hot_air_balloon', 'mosquito', 'police_car', 'sleeping_bag', 'telephone'
        , 'blackberry', 'clarinet', 'fence', 'hot_dog', 'motorbike', 'pond', 'smiley_face', 'television'
        , 'blueberry', 'clock', 'finger', 'hot_tub', 'mountain', 'pool', 'snail', 'tennis_racquet'
        , 'book', 'cloud', 'fire_hydrant', 'hourglass', 'mouse', 'popsicle', 'snake', 'tent'],
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,
    'domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
    'num_class': 345,
    'src_domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
    'tgt_domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
}

TerraIncognitaOpt = {
    'name': 'terraincognita',
    'batch_size': 32,
    'learning_rate': 0.00005,
    'file_path': './dataset/Terra',
    'classes': ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon', 'squirrel'],
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,
    'domains': ['location_100', 'location_38', 'location_43', 'location_46'],
    'num_class': 10,
    'src_domains': ['location_100', 'location_38', 'location_43', 'location_46'],
    'tgt_domains': ['location_100', 'location_38', 'location_43', 'location_46'],
}

CMNISTOpt = {
    'name': 'cmnist',
    'batch_size': 64,
    'learning_rate': 0.1,
    'file_path': './dataset/MNIST/raw',
    'classes': ['red', 'green'],
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 28,  # 2*28*28
    'domains': ['0.1', '0.2', '0.9'],
    'num_class': 2,
    'src_domains': ['0.1', '0.2', '0.9'],
    'tgt_domains': ['0.1', '0.2', '0.9'],
}

RMNISTOpt = {
    'name': 'rmnist',
    'batch_size': 64,
    'learning_rate': 0.1,
    'file_path': './dataset/MNIST/raw',
    'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 28,  # 1*28*28
    'domains': ['0', '15', '30', '45', '60', '75'],
    'num_class': 10,
    'src_domains': ['0'],
    'tgt_domains': ['15', '30', '45', '60', '75'],
}

CIFAR10Opt = {
    'name': 'cifar10',
    'batch_size': 128,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-10-C',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 10,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
                    "gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}

CIFAR100Opt = {
    'name': 'cifar100',
    'batch_size': 128,

    'learning_rate': 0.1, #initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-100-C',
    'classes': ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                'bottles', 'bowls', 'cans', 'cups', 'plates',
                'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                'bed', 'chair', 'couch', 'table', 'wardrobe',
                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                'bear', 'leopard', 'lion', 'tiger', 'wolf',
                'bridge', 'castle', 'house', 'road', 'skyscraper',
                'cloud', 'forest', 'mountain', 'plain', 'sea',
                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                'crab', 'lobster', 'snail', 'spider', 'worm',
                'baby', 'boy', 'girl', 'man', 'woman',
                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                'maple', 'oak', 'palm', 'pine', 'willow',
                'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'num_class': 100,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
        "gaussian_noise-5",
        "shot_noise-5",
        "impulse_noise-5",
        "defocus_blur-5",
        "glass_blur-5",
        "motion_blur-5",
        "zoom_blur-5",
        "snow-5",
        "frost-5",
        "fog-5",
        "brightness-5",
        "contrast-5",
        "elastic_transform-5",
        "pixelate-5",
        "jpeg_compression-5",

    ],
}

TinyImageNetOpt = {
    'name': 'tinyimagenet',
    'batch_size': 128,

    'learning_rate': 0.01,
    'weight_decay': 0.0001,
    'momentum': 0.9,

    'file_path': './dataset/Tiny-ImageNet-C',
    'classes': [i for i in range(200)],
    'num_class': 200,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
        "gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",
        # "gaussian_noise-1",
        #             "shot_noise-1",
        #             "impulse_noise-1",
        #             "defocus_blur-1",
        #             "glass_blur-1",
        #             "motion_blur-1",
        #             "zoom_blur-1",
        #             "snow-1",
        #             "frost-1",
        #             "fog-1",
        #             "brightness-1",
        #             "contrast-1",
        #             "elastic_transform-1",
        #             "pixelate-1",
        #             "jpeg_compression-1",

    ],
}

VLCSOpt = {
    'name': 'vlcs',
    'batch_size': 32,

    'learning_rate': 0.00005,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './dataset/VLCS',

    'classes': ['bird', 'car', 'chair', 'dog', 'person'],
    # 'sub_classes': ['Car', 'Pedestrian', 'Cyclist'],
    'num_class': 5,  # 8 #TODO: need to change config path as well
    # 'config_path': 'config/yolov3-kitti.cfg',
    'domains': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
    'src_domains': ['SUN09', 'LabelMe', 'VOC2007', 'Caltech101'],
    'tgt_domains': ['SUN09', 'LabelMe', 'VOC2007', 'Caltech101'],
    # 'tgt_domains': ['SUN09'],
    # 'val_domains': ['rain-200-val'],
}

PACSOpt = {
    'name': 'pacs',
    'batch_size': 32,

    'learning_rate': 0.00005,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './dataset/PACS',

    'classes': ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'],
    # 'sub_classes': ['Car', ''Pedestrian'', 'Cyclist'],
    'num_class': 7,  # 8 #TODO: need to change config path as well
    # 'config_path': 'config/yolov3-kitti.cfg',
    'domains': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'src_domains': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'tgt_domains': ['art_painting', 'cartoon', 'photo', 'sketch'],
    # 'val_domains': ['rain-200-val'],
}

OfficeHomeOpt = {
    'name': 'officehome',
    # 'batch_size': 32,
    'batch_size': 32,

    'learning_rate': 0.00005,
    # 'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './dataset/OfficeHomeDataset',

    'classes': ['Alarm_Clock', 'Bottle', 'Chair', 'Desk_Lamp', 'File_Cabinet', 'Glasses', 'Knives', 'Mop', 'Pan',
                'Printer', 'Scissors', 'Soda', 'Telephone',
                'Backpack', 'Bucket', 'Clipboards', 'Drill', 'Flipflops', 'Hammer', 'Lamp_Shade', 'Mouse', 'Paper_Clip',
                'Push_Pin', 'Screwdriver', 'Speaker', 'ToothBrush',
                'Batteries', 'Calculator', 'Computer', 'Eraser', 'Flowers', 'Helmet', 'Laptop', 'Mug', 'Pen', 'Radio',
                'Shelf', 'Spoon', 'Toys',
                'Bed', 'Calendar', 'Couch', 'Exit_Sign', 'Folder', 'Kettle', 'Marker', 'Notebook', 'Pencil',
                'Refrigerator', 'Sink', 'TV', 'Trash_Can',
                'Bike', 'Candles', 'Curtains', 'Fan', 'Fork', 'Keyboard', 'Monitor', 'Oven', 'Postit_Notes', 'Ruler',
                'Sneakers', 'Table', 'Webcam'],
    # 'sub_classes': ['Car', ''Pedestrian'', 'Cyclist'],
    'num_class': 65,  # 8 #TODO: need to change config path as well
    # 'config_path': 'config/yolov3-kitti.cfg',
    'domains': ['Art', 'Clipart', 'RealWorld', 'Product'],
    'src_domains': ['Art', 'Clipart', 'RealWorld', 'Product'],
    'tgt_domains': ['Art', 'Clipart', 'RealWorld', 'Product'],
    # 'val_domains': ['rain-200-val'],
}

SVHNOpt = {
    'name': 'svhn',
    'batch_size': 128,

    'learning_rate': 0.1, #initial learning rate
    'weight_decay': 0,
    'momentum': 0.9,

    'file_path': './dataset/svhn',
    'mnist_m_file_path': './dataset/mnist_m/test',

    'classes': ['0','1','2','3','4','5','6','7','8','9'],
    # 'sub_classes': ['Car', ''Pedestrian'', 'Cyclist'],
    'num_class': 10,
    'domains': ['SVHN', 'MNIST', 'USPS', 'MNIST-M'],
    'src_domains': ['SVHN'],
    'tgt_domains': ['MNIST', 'USPS', 'MNIST-M'],
}

VisDAOpt = {
    'name': 'visda',
    'batch_size': 128,

    'learning_rate': 0.0001, #initial learning rate
    'weight_decay': 0,
    'momentum': 0.9,

    'file_path': './dataset/VisDA',

    'classes': ['0','1','2','3','4','5','6','7','8','9','10','11'],
    # 'sub_classes': ['Car', ''Pedestrian'', 'Cyclist'],
    'num_class': 12,
    'domains': ['sim', 'real'],
    'src_domains': ['sim'],
    'tgt_domains': ['real'],
}

KITTI_SOT_Opt = {
    'name': 'kitti_sot',
    'batch_size': 64,

    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,

    'file_path': './dataset/kitti_sot',

    'classes': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person', 'Cyclist', 'Tram', 'Misc'],
    'sub_classes': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person', 'Cyclist', 'Tram', 'Misc'],
    # 'sub_classes': ['Car', 'Pedestrian', 'Cyclist'],
    'num_class': 8,
    'domains': ['2d_detection', 'original', 'rain-100mm', 'rain-100mm'],
    # 'domains': ['half1', 'half2'],
    # 'src_domains': ['half1'],
    # 'tgt_domains': ['half2'],

    'src_domains': ['2d_detection'],
    # 'src_domains': ['original'],
    # 'src_domains': ['original-val'],

    # 'tgt_domains': ['rain-200-tgt'],
    'tgt_domains': ['rain-200'],
    'val_domains': ['rain-200-val'],
    # 'src_domains': ['rain'],
    # 'tgt_domains': ['original'],
}

HARTHOpt = {
    'name': 'hhar',
    'batch_size': 64,
    'seq_len': 50,  # 128, 32, 5
    'input_dim': 3,  # 161, #6
    # 'learning_rate': 0.0001,
    # 'weight_decay': 0,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,

    'momentum': 0.9,
    # 'file_path': './dataset/harth_std_scaling_all_win32.csv', # 32, 64, 128
    # 'file_path': './dataset/harth_minmax_all_win250.csv',
    # 'file_path': './dataset/harth_minmax_all_win50.csv',
    'file_path': './dataset/harth_minmax_scaling_all_split_win50.csv',

    'classes': ['walking', 'running', 'shuffling', 'stairs ascending', 'stairs descending', 'standing', 'sitting',
                'lying', 'cycling sit', 'cycling stand', 'transport sit', 'transport stand'],
    'num_class': 12,

    # 22 users
    'users': ['S006', 'S008', 'S009', 'S010', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018', 'S019', 'S020',
              'S021', 'S022', 'S023', 'S024', 'S025', 'S026', 'S027', 'S028', 'S029'],

    'src_domains': ['S006_back', 'S009_back', 'S010_back', 'S012_back', 'S013_back', 'S014_back', 'S015_back', 'S016_back', 'S017_back', 'S020_back', 'S023_back', 'S024_back',
                    'S025_back', 'S026_back', 'S027_back'],
    'tgt_domains': [
        'S008_thigh',
        'S018_thigh',
        'S019_thigh',
        'S021_thigh',
        'S022_thigh',
        'S028_thigh',
        'S029_thigh'
    ],
}

ExtraSensoryOpt = {
    'name': 'extrasensory',
    'batch_size': 64,
    'seq_len': 5,
    'input_dim': 31,
    # 'learning_rate': 0.001,
    # 'weight_decay': 0,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'file_path': './dataset/extrasensory_selectedfeat_woutloc_std_scaling_all_win5.csv',  # 5, 10

    'classes': [
        'label:LYING_DOWN',
        'label:SITTING',
        'label:FIX_walking',
        'label:FIX_running',
        'label:OR_standing'],

    'num_class': 5,

    # 23
    'users': [
        '098A72A5-E3E5-4F54-A152-BBDA0DF7B694',
        '0A986513-7828-4D53-AA1F-E02D6DF9561B',
        '1155FF54-63D3-4AB2-9863-8385D0BD0A13',
        '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842',
        '4FC32141-E888-4BFF-8804-12559A491D8C',
        '5119D0F8-FCA8-4184-A4EB-19421A40DE0D',
        '59818CD2-24D7-4D32-B133-24C2FE3801E5',
        '61976C24-1C50-4355-9C49-AAE44A7D09F6',
        '665514DE-49DC-421F-8DCB-145D0B2609AD',
        '74B86067-5D4B-43CF-82CF-341B76BEA0F4',
        '797D145F-3858-4A7F-A7C2-A4EB721E133C',
        '7CE37510-56D0-4120-A1CF-0E23351428D2',
        '806289BC-AD52-4CC1-806C-0CDB14D65EB6',
        '9DC38D04-E82E-4F29-AB52-B476535226F2',
        'A5A30F76-581E-4757-97A2-957553A2C6AA',
        'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96',
        'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A',
        'B09E373F-8A54-44C8-895B-0039390B859F',
        'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C',
        'B9724848-C7E2-45F4-9B3F-A1F38D864495',
        'C48CE857-A0DD-4DDB-BEA5-3A25449B2153',
        'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC',
        'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B',
    ],

    'src_domains': ['098A72A5-E3E5-4F54-A152-BBDA0DF7B694',
                    '0A986513-7828-4D53-AA1F-E02D6DF9561B',
                    '1155FF54-63D3-4AB2-9863-8385D0BD0A13',
                    '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842',
                    '5119D0F8-FCA8-4184-A4EB-19421A40DE0D',
                    '665514DE-49DC-421F-8DCB-145D0B2609AD',
                    '74B86067-5D4B-43CF-82CF-341B76BEA0F4',
                    '7CE37510-56D0-4120-A1CF-0E23351428D2',
                    '806289BC-AD52-4CC1-806C-0CDB14D65EB6',
                    '9DC38D04-E82E-4F29-AB52-B476535226F2',
                    'A5A30F76-581E-4757-97A2-957553A2C6AA',
                    'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A',
                    'B09E373F-8A54-44C8-895B-0039390B859F',
                    'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C',
                    'B9724848-C7E2-45F4-9B3F-A1F38D864495',
                    'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC'],

    'tgt_domains': ['4FC32141-E888-4BFF-8804-12559A491D8C',
                    '59818CD2-24D7-4D32-B133-24C2FE3801E5',
                    '61976C24-1C50-4355-9C49-AAE44A7D09F6',
                    '797D145F-3858-4A7F-A7C2-A4EB721E133C',
                    'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96',
                    'C48CE857-A0DD-4DDB-BEA5-3A25449B2153',
                    'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B'],

}

RealLifeHAROpt = {
    'name': 'hhar',
    'batch_size': 64,

    'seq_len': 400,  # 150, 50, 5, 400
    # 'seq_len': 5,  # 150, 50, 5, 400

    'input_dim': 3,
    # 'input_dim': 54, #6, 54, 72, 90
    # 'input_dim': 72, #6, 54, 72, 90
    # 'input_dim': 90, #6, 54, 72, 90,
    # 'learning_rate': 0.0001,
    # 'weight_decay': 0,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,

    # 'file_path': './dataset/reallifehar_acc_std_scaling_all_win50.csv', #50, 100, 150

    # 'file_path': './dataset/reallifehar_acc_gps_std_scaling_win20s_0.0_0.csv',  # 54 feats
    # 'file_path': './dataset/reallifehar_acc_magn_gps_std_scaling_win20s_0.0_1.csv', #72feats
    # 'file_path': './dataset/reallifehar_acc_gyro_magn_gps_std_scaling_win20s_0.0_2.csv', #90 feats

    # 'file_path': './dataset/reallifehar_acc_gps_std_scaling_win20s_19.0_0.csv',  # overlapping 19s, 54 feats
    # 'file_path': './dataset/reallifehar_acc_magn_gps_std_scaling_win20s_19.0_1.csv', #overlapping 19s, 72feats
    # 'file_path': './dataset/reallifehar_acc_gyro_magn_gps_std_scaling_win20s_19.0_2.csv', #overlapping 19s, 90 feats

    # 'file_path': './dataset/reallifehar_acc_minmax_scaling_all_win400_overlap380.csv',
    # 'file_path': './dataset/reallifehar_acc_std_scaling_all_win400_overlap380.csv',

    'file_path': './dataset/reallifehar_acc_minmax_scaling_all_win400_overlap0.csv',
    # 'file_path': './dataset/reallifehar_acc_std_scaling_all_win400_overlap0.csv',

    'classes': ['Inactive', 'Active', 'Walking', 'Driving'],
    'num_class': 4,

    # 19
    # 'users': ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18'], # original
    # 'users': [ 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p17', 'p18'], #./dataset/reallifehar_acc_gps_win20s_0.0_0.csv
    # 'users': [ 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p17'], #./dataset/reallifehar_acc_magn_gps_win20s_0.0_1.csv
    # 'users': [ 'p0', 'p1', 'p2', 'p3', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13',], #./dataset/reallifehar_acc_gyro_magn_gps_win20s_0.0_2.csv

    # 'users': ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p17', 'p18'] # original - p15 (p15 has only 1 sample..) - p16 (p16 has only 8 samples..)

    'users': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p17', 'p18'],
    # original -p0, - p15 (p15 has only 1 sample..) - p16 (p16 has only 8 samples..)

    'src_domains': ['p1', 'p10', 'p11', 'p13', 'p14', 'p17', 'p18', 'p3', 'p4', 'p5', 'p8'],
    'tgt_domains': ['p12', 'p2', 'p6', 'p7', 'p9'],

}

ImageNetOpt = {
    # referred to for hyperparams: https://github.com/Lornatang/ResNet-PyTorch/blob/9e529757ce0607aafeae2ddd97142201b3d4cadd/examples/imagenet/main.py
    'name': 'imagenet',
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/ImageNet-C',
    # 'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 1000,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],

    'src_domains': ["original"],
    'tgt_domains': ["gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}

MNISTOpt = {
    'name': 'mnist',
    'batch_size': 128,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/MNIST-C',
    'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'num_class': 10,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "shot_noise",
                "impulse_noise",
                "glass_blur",
                "motion_blur",
                "shear",
                "scale",
                "rotate",
                "brightness",
                "translate",
                "stripe",
                "fog",
                "spatter",
                "dotted_line",
                "zigzag",
                "canny_edges",

                ],
    'src_domains': ["original"],
    'tgt_domains': [
        "shot_noise",
        "impulse_noise",
        "glass_blur",
        "motion_blur",
        "shear",
        "scale",
        "rotate",
        "brightness",
        "translate",
        "stripe",
        "fog",
        "spatter",
        "dotted_line",
        "zigzag",
        "canny_edges",
    ],

}

IMDBOpt = {
    'name': 'imdb',
    'batch_size': 16,

    'learning_rate': 0.00001,  # initial learning rate # 1e-5
    'weight_decay': 0,  # use default
    'eps': 0.00000001,  # 1e-8

    'file_path': './imdb_dataset', #'./dataset/imdb',
    'classes': ['0', '1'],
    'num_class': 2,
    'domains': ["train", "test"],
    'src_domains': ["train"],
    'tgt_domains': ["test"],
}

SST2Opt = {
    'name': 'sst-2',
    'batch_size': 16,

    'learning_rate': 0.00001,  # initial learning rate # 1e-5
    'prompt_learning_rate': 0.3,
    'weight_decay': 0,  # use default
    'eps': 0.00000001,  # 1e-8

    'file_path': './dataset/sst-2',
    'classes': ['0', '1'],
    'num_class': 2,
    'domains': ["train", "test"],
    'src_domains': ["train"],
    'tgt_domains': ["test"],
}

FineFoodOpt = {
    'name': 'finefood',
    'batch_size': 16,

    'learning_rate': 0.00001,  # initial learning rate # 1e-5
    'prompt_learning_rate': 0.3,
    'weight_decay': 0,  # use default
    'eps': 0.00000001,  # 1e-8

    'file_path': './dataset/finefood',
    'classes': ['0', '1'],
    'num_class': 2,
    'domains': ["train", "test"],
    'src_domains': ["train"],
    'tgt_domains': ["test"],
}

TomatoesOpt = {
    'name': 'tomatoes',
    'batch_size': 16,

    'learning_rate': 0.00001,  # initial learning rate # 1e-5
    'prompt_learning_rate': 0.3,
    'weight_decay': 0,  # use default
    'eps': 0.00000001,  # 1e-8

    'file_path': './dataset/tomatoes',
    'classes': ['0', '1'],
    'num_class': 2,
    'domains': ["train", "test"],
    'src_domains': ["train"],
    'tgt_domains': ["test"],
}

# processed_domains = list(np.random.permutation(args.opt['domains'], size=args.opt['num_src']))

def init_domains():
    seed = 0
    import random
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    import math

    test_size = math.ceil(len(HARTHOpt['users']) * 0.3)
    HARTHOpt['tgt_domains'] = sorted(list(np.random.permutation(HARTHOpt['users'])[:test_size]))
    HARTHOpt['src_domains'] = sorted(list(set(HARTHOpt['users']) - set(HARTHOpt['tgt_domains'])))

    test_size = math.ceil(len(ExtraSensoryOpt['users']) * 0.3)
    ExtraSensoryOpt['tgt_domains'] = sorted(list(np.random.permutation(ExtraSensoryOpt['users'])[:test_size]))
    ExtraSensoryOpt['src_domains'] = sorted(list(set(ExtraSensoryOpt['users']) - set(ExtraSensoryOpt['tgt_domains'])))

    test_size = math.ceil(len(RealLifeHAROpt['users']) * 0.3)
    RealLifeHAROpt['tgt_domains'] = sorted(list(np.random.permutation(RealLifeHAROpt['users'])[:test_size]))
    RealLifeHAROpt['src_domains'] = sorted(list(set(RealLifeHAROpt['users']) - set(RealLifeHAROpt['tgt_domains'])))

    # 15 ['S006', 'S009', 'S010', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S020', 'S023', 'S024', 'S025', 'S026', 'S027']
    # 7 ['S008', 'S018', 'S019', 'S021', 'S022', 'S028', 'S029']

    # 16 ['098A72A5-E3E5-4F54-A152-BBDA0DF7B694', '0A986513-7828-4D53-AA1F-E02D6DF9561B', '1155FF54-63D3-4AB2-9863-8385D0BD0A13', '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842', '5119D0F8-FCA8-4184-A4EB-19421A40DE0D', '665514DE-49DC-421F-8DCB-145D0B2609AD', '74B86067-5D4B-43CF-82CF-341B76BEA0F4', '7CE37510-56D0-4120-A1CF-0E23351428D2', '806289BC-AD52-4CC1-806C-0CDB14D65EB6', '9DC38D04-E82E-4F29-AB52-B476535226F2', 'A5A30F76-581E-4757-97A2-957553A2C6AA', 'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A', 'B09E373F-8A54-44C8-895B-0039390B859F', 'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C', 'B9724848-C7E2-45F4-9B3F-A1F38D864495', 'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC']
    # 7 ['4FC32141-E888-4BFF-8804-12559A491D8C', '59818CD2-24D7-4D32-B133-24C2FE3801E5', '61976C24-1C50-4355-9C49-AAE44A7D09F6', '797D145F-3858-4A7F-A7C2-A4EB721E133C', 'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96', 'C48CE857-A0DD-4DDB-BEA5-3A25449B2153', 'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B']

    # 11 ['p1', 'p10', 'p11', 'p13', 'p14', 'p17', 'p18', 'p3', 'p4', 'p5', 'p8']
    # 5 ['p12', 'p2', 'p6', 'p7', 'p9']

    print(len(HARTHOpt['src_domains']), HARTHOpt['src_domains'])
    print(len(HARTHOpt['tgt_domains']), HARTHOpt['tgt_domains'])

    print(len(ExtraSensoryOpt['src_domains']), ExtraSensoryOpt['src_domains'])
    print(len(ExtraSensoryOpt['tgt_domains']), ExtraSensoryOpt['tgt_domains'])

    print(len(RealLifeHAROpt['src_domains']), RealLifeHAROpt['src_domains'])
    print(len(RealLifeHAROpt['tgt_domains']), RealLifeHAROpt['tgt_domains'])


if __name__ == "__main__":
    init_domains()
