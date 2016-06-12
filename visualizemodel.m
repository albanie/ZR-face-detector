function visualizemodel(model, compid, name)

if nargin<2
    compid = 1:length(model.components);
end

pad = 2;
bs = 20;

for i = compid
    c = model.components{i};
    numparts = length(c);
    
    Nmix = zeros(1,numparts);
    for k = 1:numparts
        Nmix(k) = length(c(k).filterid);
    end
    
    for k = 2:numparts
        part = c(k);
        anchor = zeros(Nmix(k),2);
        for j = 1:Nmix(k)
            def = model.defs(part.defid(j));
            anchor(j,:) = [def.anchor(1) def.anchor(2)];
        end
    end
    
    part = c(1);
    % part filter
    w = model.filters(part.filterid(1)).w;
    w = foldHOG(w);
    scale = max(abs(w(:)));
    hogPicture = HOGpicture(w, bs);
    hogPicture = padarray(hogPicture, [pad pad], 0);
    hogPicture = uint8(hogPicture*(255/scale));
    % border
    hogPicture(:,1:2*pad) = 128;
    hogPicture(:,end-2*pad+1:end) = 128;
    hogPicture(1:2*pad,:) = 128;
    hogPicture(end-2*pad+1:end,:) = 128;
    im = hogPicture;
    startpoint = zeros(numparts,2);
    startpoint(1,:) = [0 0];
    
    partsize = zeros(numparts,1);
    partsize(1) = size(hogPicture,1);
    
    for k = 2:numparts
        part = c(k);
        parent = c(k).parent;
        
        % part filter
        w = model.filters(part.filterid(1)).w;
        w = foldHOG(w);
        scale = max(abs(w(:)));
        hogPicture = HOGpicture(w, bs);
        hogPicture = padarray(hogPicture, [pad pad], 0);
        hogPicture = uint8(hogPicture*(255/scale));
        % border
        hogPicture(:,1:2*pad) = 128;
        hogPicture(:,end-2*pad+1:end) = 128;
        hogPicture(1:2*pad,:) = 128;
        hogPicture(end-2*pad+1:end,:) = 128;
        
        % paste into root
        def = model.defs(part.defid(1));
        
        x1 = (def.anchor(1)-1)*bs+1 + startpoint(parent,1);
        y1 = (def.anchor(2)-1)*bs+1 + startpoint(parent,2);
        
        [H W] = size(im);
        imnew = zeros(H + max(0,1-y1), W + max(0,1-x1));
        imnew(1+max(0,1-y1):H+max(0,1-y1),1+max(0,1-x1):W+max(0,1-x1)) = im;
        im = imnew;
        
        startpoint = startpoint + repmat([max(0,1-x1) max(0,1-y1)],[numparts,1]);
        
        x1 = max(1,x1);
        y1 = max(1,y1);
        x2 = x1 + size(hogPicture,2)-1;
        y2 = y1 + size(hogPicture,1)-1;
        
        startpoint(k,1) = x1 - 1;
        startpoint(k,2) = y1 - 1;
        
        im(y1:y2, x1:x2) = hogPicture;
        partsize(k) = size(hogPicture,1);
    end
    
    % plot parts
    figure,imagesc(im);
    colormap gray;
    axis equal; axis off;
    drawnow;
    title(sprintf('Component %d',i));
    hold on;
    
    % label the first landmark
    landmarkIdx = 1;
    startX = startpoint(landmarkIdx, 1) + (partsize(landmarkIdx) / 2);
    startY = startpoint(landmarkIdx, 2) + (partsize(landmarkIdx) / 2);
    text(startX, startY, num2str(k), 'color', 'yellow', ...
            'fontSize', 18);
    
    % draw lines connecting the parts of the component, and 
    % label them by their part number
    for k = 2:numparts
        
        % find the part number of the parent of the current part
        parent = c(k).parent;
        
        % get the start and end x coordinates of the 
        % line connecting the current part and its parent
        startX = startpoint(k, 1) + (partsize(k) / 2);
        endX = startpoint(parent,1) + (partsize(parent) / 2);
        
        % similarly, get the start and end y coordinates
        startY = startpoint(k, 2) + (partsize(k) / 2);
        endY = startpoint(parent, 2) + (partsize(parent) / 2);
        
        % draw the connection
        plot([startX endX], [startY endY], 'red', ...
            'linewidth', 2);
        
        % label the landmark with its index
        text(startX, startY, num2str(k), 'color', 'yellow', ...
            'fontSize', 18);
        
    end 
    
    % save the figure as an image
    print(sprintf('figs/visualization-%s-%d', name, i), '-djpeg');
end


function f = foldHOG(w)
% f = foldHOG(w)
% Condense HOG features into one orientation histogram.
% Used for displaying a feature.

f=max(w(:,:,1:9),0)+max(w(:,:,10:18),0)+max(w(:,:,19:27),0);

function im = HOGpicture(w, bs)
% HOGpicture(w, bs)
% Make picture of positive HOG weights.

% construct a "glyph" for each orientaion
bim1 = zeros(bs, bs);
bim1(:,round(bs/2):round(bs/2)+1) = 1;
bim = zeros([size(bim1) 9]);
bim(:,:,1) = bim1;
for i = 2:9,
    bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
end

% make pictures of positive weights bs adding up weighted glyphs
s = size(w);
w(w < 0) = 0;
im = zeros(bs*s(1), bs*s(2));
for i = 1:s(1),
    iis = (i-1)*bs+1:i*bs;
    for j = 1:s(2),
        jjs = (j-1)*bs+1:j*bs;
        for k = 1:9,
            im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * w(i,j,k);
        end
    end
end
