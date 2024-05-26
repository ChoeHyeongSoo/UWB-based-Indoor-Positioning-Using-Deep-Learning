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

% 정답 레이블 좌표 파일에서 읽어오기 (labels.csv 파일이 현재 작업 디렉토리에 있어야 합니다)
labels_table = readtable('filtered_coor.csv');
labels = table2array(labels_table(:, 1:2)); % x와 y 열만 추출

% 최단 거리 점을 찾는 함수
function closest_point = find_closest_point_on_line(segment, point)
    p1 = segment(1:2);
    p2 = segment(3:4);
    
    % 선분 p1-p2와 점 point 간의 거리 계산
    v = p2 - p1;
    u = point - p1;
    t = dot(u, v) / dot(v, v);
    t = max(0, min(1, t)); % t를 0과 1 사이로 제한
    
    % 선분 위의 최단 거리 점 계산
    closest_point = p1 + t * v;
end

% 각 레이블 좌표에 대해 최단 거리 점 찾기
matching_points = zeros(size(labels));

for i = 1:size(labels, 1)
    label = labels(i, :);
    min_distance = inf;
    best_point = [0, 0];
    
    for j = 1:size(rectangle_lines, 1)
        segment = rectangle_lines(j, :);
        closest_point = find_closest_point_on_line(segment, label);
        distance = norm(label - closest_point);
        
        if distance < min_distance
            min_distance = distance;
            best_point = closest_point;
        end
    end
    
    matching_points(i, :) = best_point;
end

% 결과 출력
disp('Original Labels:');
disp(labels);
disp('Matched Points:');
disp(matching_points);

% 시각화
figure;
hold on;
plot(rectangle_points([1:end, 1], 1), rectangle_points([1:end, 1], 2), '-o'); % 사각형 그리기
plot(labels(:, 1), labels(:, 2), 'rx'); % 원래 레이블 점
plot(matching_points(:, 1), matching_points(:, 2), 'bx'); % 매칭된 점

for i = 1:size(labels, 1)
    plot([labels(i, 1), matching_points(i, 1)], [labels(i, 2), matching_points(i, 2)], 'k--'); % 매칭 선 그리기
end

hold off;
axis equal;
legend('Rectangle', 'Original Labels', 'Matched Points', 'Matching Lines');
title('Label Points Matching to Rectangle');

% 매칭된 포인트를 CSV 파일로 저장
output_table = array2table(matching_points, 'VariableNames', {'Matched_x', 'Matched_y'});
writetable(output_table, 'groundtruth.csv');

disp('Matched points saved to groundtruth.csv');