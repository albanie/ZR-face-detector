function showboxes(im, boxes, posemap)

% showboxes(im, boxes)
% Draw boxes on top of image.

imagesc(im);
hold on;
axis image;
axis off;

for box = boxes
    partsize = box.xy(1, 3) - box.xy(1, 1) + 1;
    tx = (min(box.xy(:,1)) + max(box.xy(:, 3))) / 2;
    ty = min(box.xy(:,2)) - partsize / 2;
    poseAngle = num2str(posemap(box.componentIdx));
    text(tx,ty, ...
        poseAngle, ...
        'fontsize', 18, ...
        'color','c');
    for i = size(box.xy,1):-1:1;
        x1 = box.xy(i,1);
        y1 = box.xy(i,2);
        x2 = box.xy(i,3);
        y2 = box.xy(i,4);
        line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', ...
            'color', 'b', 'linewidth', 1);
        
        plot((x1 + x2)/2, (y1 + y2)/2, 'r.', ...
            'markersize',15);
    end
end
drawnow;
