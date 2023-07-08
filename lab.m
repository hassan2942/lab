% img = imread("/MATLAB Drive/bb.jpg");
% subplot(2,2,1);
% imshow(img);
% title("normal img")
% 
% B_img = img + 50 ;
% subplot(2,2,2);
% imshow(B_img);
% title("50% brightend img")
% 
% gamma = 2.5 ;
% gamma_correction = uint8((double(B_img)/255).^gamma * 255);
% subplot(2,2,3);
% imshow(gamma_correction);
% title("50 Contrast");


% % img = imread("/MATLAB Drive/bb.jpg");

% % gray = rgb2gray(img);
% % 
% % for K = 1:8
% %     subplot(2,4,K);
% %     imshow(gray,[0 2^K-1]);
% %     title(sprintf('K = %d', K));
% % 
% % end




% Load the image
% image = imread('bb.jpg');
% 
% % Negation transform
% negation_image = 255 - image;
% 
% % Logarithmic transform
% log_transformed_image = uint8(log(double(image) + 1) / log(256) * 255);
% 
% % Power transform
% gamma = 0.5;
% power_transformed_image = uint8((double(image) / 255).^gamma * 255);
% 
% % Display the original and transformed images
% subplot(2, 2, 1);
% imshow(image);
% title('Original Image');
% 
% subplot(2, 2, 2);
% imshow(negation_image);
% title('Negation Transform');
% 
% subplot(2, 2, 3);
% imshow(log_transformed_image);
% title('Logarithmic Transform');
% 
% subplot(2, 2, 4);
% imshow(power_transformed_image);
% title('Power Transform');



% i =  imread("flower.jfif");
% gray = rgb2gray(i);
% 
% subplot(2,2,1);
% imshow(gray);
% title("gray")
% 
% 
% sobel = edge(gray,"sobel");
% subplot(2,2,2)
% imshow(sobel);
% title("sobel");
% 
% 
% prewitt = edge(gray,"prewitt");
% subplot(2,2,3);
% imshow(prewitt);
% title("Prewitt")
% 
% 
% canny =  edge(gray,"canny");
% subplot(2,2,4);
% imshow(canny);
% title("canny");

% 
% RGB = imread("round.jpg");
% I = im2gray(RGB);
% bw = imbinarize(I);
% minSize = 30;
% bw = bwareaopen(bw,minSize);
% se = strel("disk",2);
% bw = imclose(bw,se);
% imshow(bw);
% bw = imfill(bw,"holes");
% imshow(bw);
% [B,L] = bwboundaries(bw,"noholes");
% imshow(label2rgb(L,@jet,[.5 .5 .5]))
% hold on
% for k = 1:length(B)
%   boundary = B{k};
%   plot(boundary(:,2),boundary(:,1),"w",LineWidth=2)
% end
% title("Objects with Boundaries in White");
% stats = regionprops(L,"Circularity","Centroid");
% threshold = 0.94;
% 
% for k = 1:length(B)
% 
%   % Obtain (X,Y) boundary coordinates corresponding to label "k"
%   boundary = B{k};
% 
%   % Obtain the circularity corresponding to label "k"
%   circ_value = stats(k).Circularity;
% 
%   % Display the results
%   circ_string = sprintf("%2.2f",circ_value);
% 
%   % Mark objects above the threshold with a black circle
%   if circ_value > threshold
%     centroid = stats(k).Centroid;
%     plot(centroid(1),centroid(2),"ko");
%   end
% 
%   text(boundary(1,2)-35,boundary(1,1)+13,circ_string,Color="y",...
%        FontSize=14,FontWeight="bold")
% 
% end
% title("Centroids of Circular Objects and Circularity Values")
% 
% 
% 


% I = imread('bb.jpg');
% imshow(I);
% h = fspecial('log',7,0.8);
% I2 = imfilter(I,h);
% 
% 
% j = I- I2 ;
% imshow(j);

% % I = imread("abc.png");
% % imshow(I);
% % 
% % params.MinArea = 20;
% % params.MinAspectRatio = 0.062;
% % params.MaxAspectRatio = 4;
% % 
% % bboxes = helperDetectTextRegions(I, params);
% % showShape("rectangle",bboxes);



% 

S = imread('/MATLAB Drive/enhanced_image.png');

level = graythresh(S);

BW = im2bw(S,level);

subplot(3,3,6);

imshow(BW);

title('Segmentation');

image_bin = '/MATLAB Drive/bin.png';

imwrite(BW,image_bin);

BW1 = imread('bin.png');

SE = strel('rectangle',[30 20]);

BW2 = imopen(BW1, SE);

subplot(3,3,7);

imshow(BW2);

title('Structure Element');

BW3 = imerode(BW2,SE);

subplot(3,3,8);

imshow(BW3);

title('Erosion');

BW4 = imdilate(BW3,SE);

subplot(3,3,9);

imshow(BW4);

title('Dilation');

area = 1793;

eccentricity = 0.7319;

perimeter = 161.6980;

feature_matrix = [101,0.1,20 ; 1200,0.4,378; 6000,0.2,100 ; 500,0.4,150 ; 200, 0.6,200 ; 900 , 0.9 , 300];

labels = [1,1,1,0,0,0];

svm_model = fitcsvm(feature_matrix, labels);

new_tumor = [area, eccentricity, perimeter];

predicted_label = predict(svm_model, new_tumor);

if predicted_label == 1

disp("Abnormal")

else

disp("Normal")

end
% 



A = imread('lungcancer.PNG');

imhist(A)
I = imgaussfilt(A,0.1)


k=imadjust(I,[],[],1.5)

level=graythresh(k)
bw=imbinarize(rgb2gray(k),level)

SE = strel('rectangle',[40 30]);



BW3 = imerode(bw,SE);
imshow(BW3)



BW4 = imdilate(BW3,SE);
imshow(BW4)




