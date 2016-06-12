function pyramid = featpyramid(im, model)
% pyramid = featpyramid(im, model, padx, pady);
% Compute feature pyramid.
%
% pyramid.feat{i} is the i-th level of the feature pyramid.
% pyramid.scales{i} is the scaling factor used for the i-th level.
% pyramid.feat{i+interval} is computed at exactly half the resolution 
% of feat{i}.
% first octave halucinates higher resolution data.

% retrieve the interval stored with the model (by default, 5)
interval  = model.interval;

% retrieve the spatial bin size used by the HoG cells (by default, 4)
sbin = model.sbin;


%------------------------------------------
% Padding
%------------------------------------------
% Each part consists of a HoG cell of dimension `model.maxsize`
% (by default, [5 5]).

% NOTE: I'm not entirely sure what this comment means ->

% "Select padding, allowing for one cell in model to be visible
% Even padding allows for consistent spatial relations across 2X scales"
padx = max(model.maxsize(2) - 1 - 1, 0);
pady = max(model.maxsize(1) - 1 - 1, 0);

% calculate scale factor from the interval value
sc = 2 ^(1/interval);

% calculate the maximum scale used in the pyramid
imsize = [size(im, 1) size(im, 2)];

% The smallest feature layer requires 5 (the number of pixels required
% to generate a HoG cell) times the size of spatial bins used to generate
% a landmark 
minResolution = 5 * sbin;
max_scale = 1 + floor( log( min(imsize) / minResolution) / log(sc) );

% create fields to store the features and associated scales
pyramid.feat  = cell(max_scale + interval, 1);
pyramid.scale = zeros(max_scale + interval, 1);

% convert to double, the datatype required by the resize function 
im = double(im);

% loop over intervals and compute 
for i = 1:interval
    
    % resize the image to the desired scale
    % NOTE: for grayscale images, we duplicate across layers
    if numel(size(im)) == 2
        tmp = im;
        im = zeros([size(tmp) 3]);
        im(:,:,1) = tmp;
        im(:,:,2) = tmp;
        im(:,:,3) = tmp;
    end
    scaled = resize(im, 1/sc^(i-1));
    
    % compute the "first" 2x interval
    pyramid.feat{i} = features(scaled, sbin / 2);
    pyramid.scale(i) = 2 / sc^(i - 1);
    
    % compute the "second" 2x interval
    pyramid.feat{i + interval} = features(scaled, sbin);
    pyramid.scale(i + interval) = 1 / sc^(i-1);
    
    % remaining interals
    for j = i+interval:interval:max_scale
        scaled = reduce(scaled);
        pyramid.feat{j+interval} = features(scaled, sbin);
        pyramid.scale(j+interval) = 0.5 * pyramid.scale(j);
    end
end

for i = 1:length(pyramid.feat)
    % add 1 to padding because feature generation deletes a 1-cell
    % wide border around the feature map
    pyramid.feat{i} = padarray(pyramid.feat{i}, [pady+1 padx+1 0], 0);
    % write boundary occlusion feature
    pyramid.feat{i}(1:pady+1, :, end) = 1;
    pyramid.feat{i}(end-pady:end, :, end) = 1;
    pyramid.feat{i}(:, 1:padx+1, end) = 1;
    pyramid.feat{i}(:, end-padx:end, end) = 1;
end

pyramid.scale    = model.sbin./pyramid.scale;
pyramid.interval = interval;
pyramid.imy = imsize(1);
pyramid.imx = imsize(2);
pyramid.pady = pady;
pyramid.padx = padx;