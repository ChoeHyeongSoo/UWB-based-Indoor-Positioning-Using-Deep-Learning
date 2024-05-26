% 사각형 정의
rectangle_points = [...
    0.9663, 4.62724; % 점 1
    0.9663, 1.61536; % 점 2
    3.0635, 1.61536; % 점 3
    3.0635, 4.62724  % 점 4
];

% 사각형 변 계산
rectangle_lines = [...
    rectangle_points(1,:), rectangle_points(2,:);
    rectangle_points(2,:), rectangle_points(3,:);
    rectangle_points(3,:), rectangle_points(4,:);
    rectangle_points(4,:), rectangle_points(1,:);
];

% 정답 레이블 좌표 파일에서 읽어오기 (location.csv 파일이 현재 작업 디렉토리에 있어야 합니다)
labels_table = readtable('location.csv');
labels = table2array(labels_table(:, 1:2)); % x와 y 열만 추출

% 입력 레이블 데이터 파일에서 읽어오기 (input_labels.csv 파일이 현재 작업 디렉토리에 있어야 합니다)
input_labels_table = readtable('TOA.csv');
input_labels = table2array(input_labels_table(:, :)); % 모든 열 추출 (n, 4 형태)

% non_outlier_idx.mat 파일에서 논리 인덱스 불러오기
load('non_outlier_idx.mat', 'non_outliers');

% non_outliers를 사용하여 이상치 제거
filtered_labels = labels(non_outliers, :);
filtered_input_labels = input_labels(non_outliers, :);

% 결과 출력
disp('Filtered Labels:');
disp(filtered_labels);
disp('Filtered Input Labels:');
disp(filtered_input_labels);

% 시각화
figure;
hold on;
plot(labels(:, 1), labels(:, 2), 'ro'); % 원래 데이터 점 (이상치 포함)
plot(filtered_labels(:, 1), filtered_labels(:, 2), 'bx'); % 이상치 제거된 데이터
plot(rectangle_points([1:end, 1], 1), rectangle_points([1:end, 1], 2), '-o'); % 사각형 그리기
axis equal;
legend('Original Labels', 'Filtered Labels', 'Rectangle');
title('Filtered Label Points with Non-Outliers');
hold off;

% 이상치 제거된 데이터 저장
filtered_labels_table = array2table(filtered_labels, 'VariableNames', {'x', 'y'});
writetable(filtered_labels_table, 'filtered_coor.csv');

filtered_input_labels_table = array2table(filtered_input_labels, 'VariableNames', {'x1', 'y1', 'x2', 'y2'});
writetable(filtered_input_labels_table, 'filtered_distance.csv');

disp('Filtered labels and input labels saved to CSV files');
