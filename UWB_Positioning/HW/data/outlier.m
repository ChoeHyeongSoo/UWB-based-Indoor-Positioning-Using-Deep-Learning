% non_outlier_idx.mat 파일 로드
load('non_outlier_idx.mat', 'non_outliers');

% CSV 파일에서 데이터를 불러오기
input_labels_table = readtable('data/new_hw/TOA.csv');  % 입력 레이블 파일
input_labels = table2array(input_labels_table);     % 입력 레이블 배열로 변환

labels_table = readtable('data/new_hw/location.csv');            % 정답 레이블 파일
labels = table2array(labels_table(:, 1:2));          % 정답 레이블 배열로 변환

% 이상치 인덱스 추출 (필터링되지 않은 데이터)
outliers = ~non_outliers;

% 이상치 데이터 추출
filtered_outlier_input_labels = input_labels(outliers, :);
filtered_outlier_labels = labels(outliers, :);

% 이상치 데이터 출력
disp('Outlier Input Labels:');
disp(filtered_outlier_input_labels);

disp('Outlier Labels:');
disp(filtered_outlier_labels);

% 이상치 데이터 CSV 파일로 저장
outlier_input_labels_table = array2table(filtered_outlier_input_labels, 'VariableNames', {'Input1', 'Input2', 'Input3', 'Input4'});
writetable(outlier_input_labels_table, 'outlier_dist.csv');

outlier_labels_table = array2table(filtered_outlier_labels, 'VariableNames', {'x', 'y'});
writetable(outlier_labels_table, 'outlier_coor.csv');
