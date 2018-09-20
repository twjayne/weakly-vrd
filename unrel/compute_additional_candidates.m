% Run edgeboxes on the dataset in order to collect more object proposals.
% Save object proposals to 'unlabelled_boxes' dir.
% 
% Before running:
% 	export LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/4.8/libgomp.so

function compute_additional_candidates ()

	addpath('/home/markham/data/3rdparty/pdollar/toolbox-master/channels');
	addpath('/home/markham/data/3rdparty/pdollar/toolbox-master/matlab');
	addpath('/home/markham/data/3rdparty/pdollar/edges-master');

	load('/home/markham/data/3rdparty/pdollar/edges-master/models/forest/modelBsds.mat')
	load('/home/markham/data/unrel/data/vrd-dataset/image_filenames_train.mat');
	
	for idx = 1:numel(image_filenames)
		fpath = ['/home/markham/data/unrel/data/vrd-dataset/images/train/' image_filenames{idx}];
		save_bbs_for_one_img(model, idx, fpath);
	end
end

function save_bbs_for_one_img (model, idx, fpath)
	disp(idx)
	I = imread(fpath);
	unlabelled_boxes = edgeBoxes(I, model);
	outpath = ['/home/markham/data/unrel/data/vrd-dataset/train/unlabelled_boxes/' sprintf('%d.mat', idx)];
	save(outpath, 'unlabelled_boxes');
end
