function [ F ] = imgFusion( folder, opts )
%imgFusion - Implementation of the following paper
%Image Fusion with Guided Filtering[1]
%Shutao Li, Member, IEEE, Xudong Kang, Student Member, IEEE, and Jianwen Hu
%IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 22, NO. 7, JULY 2013
%F = Fused image
%folder = folder name containing the images to be fused
%opts = 1 to display result, 2 to save it into a jpeg file

%Code by Raphael Ruschel dos Santos 17/06/2017


assert(isdir(folder)==1,strcat('Folder "', folder, '" was not found'));
assert((opts < 1 || opts > 2) == 0,'Invalid option');

%Filter params as specified on [1]
avgFilterSize = 31;
gaussParams = [11 5];
guidedParams = [45 0.3;7 1E-6];

%Build filters
Z = ones(avgFilterSize,avgFilterSize)./avgFilterSize^2; %Box filter, Wsize = 31
L = [0 -1 0;-1 4 -1;0 -1 0]; %3x3 Laplacian Filter
g = fspecial('gaussian',gaussParams(1),gaussParams(2)); %Gaussian filter, Wsize = 11; std = 5

cd(folder)
d = dir();
d = d(3:end);
N = size(d,1);

for i=1:N %Read all the images on the folder
    img = imread(d(i).name);
    if(i == 1)
        imgSz = size(img);
        color = 0;
        if(size(imgSz,2) > 2) %If color image, allocate necessary resources
            color = 1;
            img_c = zeros([imgSz N]);
        end
        %Allocate resources
        img_g = zeros(imgSz(1),imgSz(2),N);
        Wb = img_g;
        Wd = img_g;
        S = img_g;
        F = zeros(imgSz);
    end
    if(color == 1)
        img_g(:,:,i) = double(rgb2gray(img));
        img_c(:,:,:,i) = double(img);
    else
        img_g(:,:,i) = img;
    end
    %Equation 12
    H = abs(conv2(img_g(:,:,i),L,'same'));
    %Equation 13
    S(:,:,i) = conv2(H,g,'same');
end

cd ..

[~,idx] = max(S,[],3);

for i=1:N
    %Equation 14
    aux = zeros(imgSz(1), imgSz(2));
    aux(idx == i) = 1;
    P = aux;
    %Equation 15 & 16
    Wb(:,:,i) = imguidedfilter(P,img_g(:,:,i),'NeighborhoodSize',[guidedParams(1,1) guidedParams(1,1)],'DegreeOfSmoothing',guidedParams(1,2));
    Wd(:,:,i) = imguidedfilter(P,img_g(:,:,i),'NeighborhoodSize',[guidedParams(2,1) guidedParams(2,1)],'DegreeOfSmoothing',guidedParams(2,2));
end

Wb = normalizeWeights(Wb,N);
Wd = normalizeWeights(Wd,N);

%B = Equation 10
%D = Equation 11
%F = Eq. 17,18 and 19
if(color ==1)
    for i=1:N
        B = convn(img_c(:,:,:,i),Z,'same');
        D = img_c(:,:,:,i) - B;
        F = F + B.*repmat(Wb(:,:,i),[1 1 3]) + D.*repmat(Wd(:,:,i),[1 1 3]);
    end
else
    for i=1:N
        B = conv2(img_g(:,:,i),Z,'same');
        D = img_g(:,:,i) - B;
        F = F + B.*Wd(:,:,i) + D.*Wd(:,:,i);
    end
end

if(opts == 1)
    imshow(uint8(F));
    title('Fused image');
elseif(opts == 2)
    cd(folder);
    imwrite(uint8(F),'fused.jpg');
    cd ..
end

end

