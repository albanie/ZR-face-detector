function boxes = detect(input, model, thresh)

% Keep track of detected boxes and features
BOXCACHESIZE = 100000;
cnt = 0;
boxes.score  = 0;
boxes.componentIdx  = 0;
boxes.xy = 0;
boxes.level = 0;
boxes(BOXCACHESIZE) = boxes;

% Compute the feature pyramid and prepare filters
pyramid = featpyramid(input, model);

[components, filters, resp]  = modelcomponents(model, pyramid);

% loop over each component of the model in a random order
for componentIdx  = randperm(length(components)),
    
    minLevel = model.interval + 1;
    levels   = minLevel : length(pyramid.feat);
    
    % loop over levels in a random order
    for rlevel = levels(randperm(length(levels))),
        
        % select the parts of the current component
        parts    = components{componentIdx};
        numparts = length(parts);
        
        % calculate the local part scores
        for k = 1:numparts
            
            % select by filter index
            filterId = parts(k).filterid;
            
            
            level = rlevel - parts(k).scale * model.interval;
            
            if isempty(resp{level}),
                resp{level} = fconv(pyramid.feat{level}, ...
                    filters, ...
                    1, ...
                    length(filters));
            end
            parts(k).score = resp{level}{filterId};
            parts(k).level = level;
            
        end
        
        % Walk from leaves to root of tree, passing message to parent
        % Given a 2D array of filter scores 'child', shiftdt() does the following:
        % (1) Apply distance transform
        % (2) Shift by anchor position (child.startxy) of part wrt parent
        % (3) Downsample by child.step
        for k = numparts:-1:2
            
            % Define the current part as the `child`
            child = parts(k);
            
            % find its parent
            parent   = child.parent;
            
            % retrieve the size of the parent part score
            [Ny, Nx, ~] = size(parts(parent).score);
            
            % apply the complex shiftdf function
            [msg,parts(k).Ix,parts(k).Iy] = shiftdt(...
                child.score, ...
                child.w(1),child.w(2),child.w(3),child.w(4), ...
                child.startx, ...
                child.starty, ...
                Nx, ...
                Ny, ...
                child.step);
            
            % update the parent part score
            parts(parent).score = parts(parent).score + msg;
        end
        
        % Add bias to root score
        rscore = parts(1).score + parts(1).w;
        
        % determine whether the root score exceeds the threshold
        [Y,X] = find(rscore >= thresh);
        
        
        if ~isempty(X)
            XY = backtrack( X, Y, parts, pyramid);
        end
        
        % Walk back down tree following pointers
        for i = 1:length(X)
            x = X(i);
            y = Y(i);
            
            if cnt == BOXCACHESIZE
                b0 = nms_face(boxes,0.3);
                clear boxes;
                boxes.score  = 0;
                boxes.componentIdx  = 0;
                boxes.xy = 0;
                boxes.level = 0;
                boxes(BOXCACHESIZE) = boxes;
                cnt = length(b0);
                boxes(1:cnt) = b0;
            end
            
            cnt = cnt + 1;
            boxes(cnt).componentIdx = componentIdx;
            boxes(cnt).score = rscore(y,x);
            boxes(cnt).level = rlevel;
            boxes(cnt).xy = XY(:,:,i);
        end
    end
end

boxes = boxes(1:cnt);



%---------------------------------------
function box = backtrack(x, y, parts, pyramid)
%---------------------------------------
% Backtrack through dynamic programming messages to estimate part 
% locations and the associated feature vector

numparts = length(parts);
ptr = zeros(numparts,2,length(x));
box = zeros(numparts,4,length(x));
k   = 1;
p   = parts(k);
ptr(k,1,:) = x;
ptr(k,2,:) = y;
% image coordinates of root
scale = pyramid.scale(p.level);
padx  = pyramid.padx;
pady  = pyramid.pady;
box(k,1,:) = (x-1-padx)*scale + 1;
box(k,2,:) = (y-1-pady)*scale + 1;
box(k,3,:) = box(k,1,:) + p.sizx*scale - 1;
box(k,4,:) = box(k,2,:) + p.sizy*scale - 1;

for k = 2:numparts,
    p   = parts(k);
    par = p.parent;
    x   = ptr(par,1,:);
    y   = ptr(par,2,:);
    inds = sub2ind(size(p.Ix), y, x);
    ptr(k,1,:) = p.Ix(inds);
    ptr(k,2,:) = p.Iy(inds);
    % image coordinates of part k
    scale = pyramid.scale(p.level);
    box(k,1,:) = (ptr(k,1,:)-1-padx)*scale + 1;
    box(k,2,:) = (ptr(k,2,:)-1-pady)*scale + 1;
    box(k,3,:) = box(k,1,:) + p.sizx*scale - 1;
    box(k,4,:) = box(k,2,:) + p.sizy*scale - 1;
end

%-------------------------------------------------------------------
function [components, filters, resp] = modelcomponents(model, pyramid)
%-------------------------------------------------------------------
% Cache various statistics from the model data structure for later use

components = cell(length(model.components),1);

% loop over model components (i.e. mixtures)
for c = 1:length(model.components)
    
    % loop over parts (i.e. facial landmarks) in each component
    for k = 1:length(model.components{c})
        
        % load part (it has been stored as part of the trained model)
        part = model.components{c}(k);
        
        % load the filters associated with that part
        partFilters = model.filters(part.filterid);
        
        % find the part filter size
        [part.sizy, part.sizx, ~] = size(partFilters.w);
        
        % attach the filter index
        part.filterI = partFilters.i;
        partFilters = model.defs(part.defid);
        part.defI = partFilters.i;
        
        % attach the weight
        part.w    = partFilters.w;
        
        % store the scale of each part relative to the component root
        parent = part.parent;
        assert(parent < k);
        anchorX  = partFilters.anchor(1);
        anchorY  = partFilters.anchor(2);
        ds  = partFilters.anchor(3);
        
        if parent > 0,
            part.scale = ds + components{c}(parent).scale;
        else
            assert(k == 1);
            part.scale = 0;
        end
        
        % amount of (virtual) padding to hallucinate
        step     = 2^ds;
        virtpady = (step - 1) * pyramid.pady;
        virtpadx = (step - 1) * pyramid.padx;
        
        % starting points (simulates additional padding at finer scales)
        part.starty = anchorY - virtpady;
        part.startx = anchorX - virtpadx;
        part.step   = step;
        part.level  = 0;
        part.score  = 0;
        part.Ix     = 0;
        part.Iy     = 0;
        components{c}(k) = part;
    end
end

% return a cellarray of model filters, together with an 
% empty cell array of the same dimensions as the pyramid features
% to hold the responses
resp    = cell(length(pyramid.feat),1);
filters = cell(length(model.filters),1);
for i = 1:length(filters),
    filters{i} = model.filters(i).w;
end

