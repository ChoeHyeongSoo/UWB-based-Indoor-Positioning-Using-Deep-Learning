% % 정답 레이블 좌표 파일에서 읽어오기 (labels.csv 파일이 현재 작업 디렉토리에 있어야 합니다)
% labels_table = readtable('location.csv');
% labels = table2array(labels_table(:, 1:2)); % x와 y 열만 추출
% 
% % 이상치 제거를 위한 임계값 설정
% threshold_distance = 1.4; % 임계값 (중심에서의 거리)
% 
% % K-means 클러스터링 수행하여 이상치 제거
% [idx, C] = kmeans(labels, 2); % 예시로 2개의 클러스터 사용
% distances = zeros(size(labels, 1), 1);
% 
% for i = 1:size(labels, 1)
%     distances(i) = norm(labels(i, :) - C(idx(i), :));
% end
% 
% non_outliers = distances < threshold_distance;
% filtered_labels = labels(non_outliers, :);
% 
% % 시각화
% figure;
% hold on;
% plot(filtered_labels(:, 1), filtered_labels(:, 2), 'rx'); % 이상치 제거된 데이터 플로팅
% axis equal;
% legend('Filtered Labels');
% title('Filtered Label Points');

% ========================

% 정답 레이블 좌표 파일에서 읽어오기 (labels.csv 파일이 현재 작업 디렉토리에 있어야 합니다)
labels_table = readtable('location.csv');
labels = table2array(labels_table(:, 1:2)); % x와 y 열만 추출

% 클러스터의 수 설정 (예: 5개의 클러스터)
num_clusters = 40;

% K-means 클러스터링 수행
[idx, C] = kmeans(labels, num_clusters);

% 각 클러스터 내의 평균 거리와 표준 편차 계산
threshold_multiplier = 2; % 표준 편차를 몇 배로 할 것인지 설정
non_outliers = true(size(labels, 1), 1);

for k = 1:num_clusters
    cluster_points = labels(idx == k, :);
    cluster_center = C(k, :);
    distances = sqrt(sum((cluster_points - cluster_center) .^ 2, 2));
    mean_distance = mean(distances);
    std_distance = std(distances);
    
    % 임계값을 벗어나는 점들을 이상치로 간주
    threshold_distance = mean_distance + threshold_multiplier * std_distance;
    non_outliers(idx == k) = distances < threshold_distance;
end

% 이상치 제거된 데이터
filtered_labels = labels(non_outliers, :);

% 결과 출력
disp('Filtered Labels:');
disp(filtered_labels);

% 시각화
figure;
hold on;
plot(labels(:, 1), labels(:, 2), 'ro'); % 원래 데이터 점 (이상치 포함)
plot(filtered_labels(:, 1), filtered_labels(:, 2), 'bx'); % 이상치 제거된 데이터
for k = 1:num_clusters
    cluster_center = C(k, :);
    plot(cluster_center(1), cluster_center(2), 'ks', 'MarkerSize', 10, 'LineWidth', 2); % 클러스터 중심
end
axis equal;
legend('Original Labels', 'Filtered Labels', 'Cluster Centers');
title('Filtered Label Points with K-means Clustering');
hold off;

% 이상치 제거된 데이터와 클러스터 인덱스를 테이블로 변환
filtered_labels_table = array2table(filtered_labels, 'VariableNames', {'x', 'y'});
filtered_labels_table.ClusterIndex = idx(non_outliers);

% CSV 파일로 저장
writetable(filtered_labels_table, 'filtered_location.csv');

disp('Filtered labels and cluster indices saved to filtered_location.csv');