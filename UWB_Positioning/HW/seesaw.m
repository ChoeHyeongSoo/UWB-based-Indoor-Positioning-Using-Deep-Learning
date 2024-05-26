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

% CSV 파일에서 데이터 읽어오기
data = readtable('data/AP/groundtruth.csv');

% 데이터가 들어있는 테이블의 첫 줄이 헤더인 경우
% % x = data.x;
% % y = data.y;
% x = data.Matched_x;
% y = data.Matched_y;

% xy 좌표 시각화
figure;
hold on;

% 데이터 포인트 그리기
scatter(x, y, 50, 'filled', 'r');

% 사각형 그리기
plot(rectangle_points([1:end, 1], 1), rectangle_points([1:end, 1], 2), '-o', 'LineWidth', 2, 'Color', 'b');

axis equal;
title('Data Points and Rectangle');
xlabel('X');
ylabel('Y');
grid on;

% 범례 추가
legend('Data Points', 'Rectangle');

hold off;

% 결과 출력
disp('X and Y coordinates:');
disp([x, y]);

% =============

% CSV 파일에서 이상치 데이터 읽어오기
outlier_data = readtable('data/AP/outlier_coor.csv');

% 이상치 데이터가 들어있는 테이블의 첫 줄이 헤더인 경우
outlier_x = outlier_data.x;
outlier_y = outlier_data.y;

% xy 좌표 시각화
figure;
hold on;

% 데이터 포인트 그리기
scatter(x, y, 50, 'filled', 'r');

% 이상치 데이터 포인트 그리기
scatter(outlier_x, outlier_y, 50, 'filled', 'b');

% 사각형 그리기
plot(rectangle_points([1:end, 1], 1), rectangle_points([1:end, 1], 2), '-o', 'LineWidth', 2, 'Color', 'g');

axis equal;
title('Data Points, Outliers, and Rectangle');
xlabel('X');
ylabel('Y');
grid on;

% 범례 추가
legend('Data Points', 'Outliers', 'Rectangle');

hold off;

% 결과 출력
disp('X and Y coordinates:');
disp([x, y]);
