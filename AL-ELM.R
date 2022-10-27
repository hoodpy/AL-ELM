matrix_vector = function(number) {
  the_vector = matrix(seq(1, by = 0, length = number), ncol = 1, nrow = number)
  return(the_vector)
}

sigmoid_function <- function(x) {
  x <- as.matrix(x)
  x_nrow <- nrow(x)
  x_ncol <- ncol(x)
  y <- matrix(nrow = x_nrow, ncol = x_ncol)
  for (i in 1:x_nrow) {
    for (j in 1:x_ncol) {
      y[i, j] <- 1 / (1 + exp(-x[i, j]))
    }
  }
  return(y)
}

I_function = function(n_row) {
  y = matrix(seq(0, by = 0, length = n_row^2), nrow = n_row, ncol = n_row)
  for (i in 1:n_row) {
    y[i, i] = 1
  }
  return(y)
}

normial <- function(x) {
  return((2 * (x - min(x)) / (max(x) - min(x))) - 1)
}

obtained_acc_G_mean = function(x) {
  the_sum = 0
  the_G_mean = 1
  for (i in 1:nrow(x)) {
    the_sum = the_sum + x[i, i]
    the_G_mean = the_G_mean * (x[i, i] / sum(x[i,]))
  }
  the_acc = the_sum / sum(x)
  the_G_mean = the_G_mean^(1 / nrow(x))
  return(list(the_acc * 100, the_G_mean * 100))
}

model = function(c, node, num_classes, train_path, samples_number = 0, test_path = "src", margin_threshold = 0.2, bound_selection = 10, single = FALSE, init_num = 100) {
  if (single) {
    total_data = read.table(train_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
  } else {
    data_train = read.table(train_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
    data_test = read.table(test_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
    total_data = rbind(data_train, data_test)
    samples_number = nrow(data_train)
  }
  variables_number = ncol(total_data) - 1
  total_data$label = as.numeric(total_data$label)
  total_data_normial = as.data.frame(lapply(total_data[, c(1:variables_number)], normial))
  total_data = cbind(total_data_normial, total_data[variables_number + 1])
  data = total_data[c(1:samples_number),]
  testing_data = total_data[-c(1:samples_number),]
  init_indic = sample(1:samples_number, sample(1:init_num, 1), replace = FALSE)
  training_data = data[init_indic,]
  while (length(unique(training_data[, variables_number + 1])) != num_classes) {
    init_indic = sample(1:samples_number, sample(1:init_num, 1), replace = FALSE)
    training_data = data[init_indic,]
  }
  data = data[-init_indic,]
  the_number_select = nrow(training_data)
  training_data_variables = as.matrix(training_data[, c(1:variables_number)])
  instances_labels = training_data[, variables_number + 1]
  categories = unique(instances_labels)
  training_data_labels = as.data.frame(matrix(seq(0, by = 0, length = nrow(training_data) * num_classes), nrow = nrow(training_data), ncol = num_classes))
  names(training_data_labels) = categories
  for (i in 1:num_classes) {
    position = which(instances_labels == categories[i])
    training_data_labels[position, i] = 1
  }
  training_data_labels = as.matrix(training_data_labels)
  input_weight = matrix(rnorm(variables_number * node, mean = 0, sd = 1), nrow = variables_number, ncol = node)
  input_bisa = matrix(runif(node, min = -1, max = 1), nrow = 1, ncol = node)
  H = sigmoid_function(training_data_variables %*% input_weight + matrix_vector(nrow(training_data)) %*% input_bisa)
  K = I_function(node) / c + t(H) %*% H
  Beta = solve(K) %*% t(H) %*% training_data_labels
  while (TRUE) {
    if (nrow(data) == 0) {
      break()
    }
    data_variables = as.matrix(data[, c(1:variables_number)])
    H = sigmoid_function(data_variables %*% input_weight + matrix_vector(nrow(data)) %*% input_bisa)
    aim = as.data.frame(H %*% Beta)
    aim_sum = apply(aim, 1, sum)
    for (i in 1:nrow(aim)) {
      aim[i,] = aim[i,] / aim_sum[i]
    }
    aim$the_margin = 1
    for (i in 1:nrow(aim)) {
      the_result_sample = sort(aim[i, c(1:(ncol(aim) - 1))], decreasing = TRUE)
      aim[i, ncol(aim)] = the_result_sample[1] - the_result_sample[2]
    }
    aim = aim[order(aim$the_margin),]
    aim_select = data.frame()
    for (i in 1:nrow(aim)) {
      if (aim[i, ncol(aim)] <= margin_threshold) {
        aim_select = rbind(aim_select, aim[i,])
      }
    }
    if (nrow(aim_select) > bound_selection) {
      aim_select = aim_select[c(1:bound_selection),]
    }
    if (nrow(aim_select) == 0) {
      break()
    }
    the_number_select = the_number_select + nrow(aim_select)
    samples_index = as.numeric(row.names(aim_select))
    selected_samples = data[samples_index, ]
    selected_samples_variables = as.matrix(selected_samples[, c(1:variables_number)])
    instances_labels = selected_samples[, variables_number + 1]
    selected_samples_labels = as.data.frame(matrix(seq(0, by = 0, length = nrow(selected_samples) * num_classes), nrow = nrow(selected_samples), ncol = num_classes))
    names(selected_samples_labels) = categories
    for (i in 1:num_classes) {
      position = which(instances_labels == categories[i])
      selected_samples_labels[position, i] = 1
    }
    selected_samples_labels = as.matrix(selected_samples_labels)
    training_data = rbind(training_data, selected_samples)
    training_data_variables = rbind(training_data_variables, selected_samples_variables)
    training_data_labels = rbind(training_data_labels, selected_samples_labels)
    data = data[-samples_index,]
    H = sigmoid_function(training_data_variables %*% input_weight + matrix_vector(nrow(training_data)) %*% input_bisa)
    K = I_function(node) / c + t(H) %*% H
    Beta = solve(K) %*% t(H) %*% training_data_labels
  }
  
  testing_data_variables = as.matrix(testing_data[, c(1:variables_number)])
  H = sigmoid_function(testing_data_variables %*% input_weight + matrix_vector(nrow(testing_data)) %*% input_bisa)
  aim = as.data.frame(H %*% Beta)
  aim_result = aim[, order(as.numeric(colnames(aim)))]
  aim_result$result = 0
  for (i in 1:nrow(aim_result)) {
    aim_result[i, ncol(aim_result)] = which.max(aim_result[i, c(1:(ncol(aim_result) - 1))])
  }
  table0 = table(testing_data$label, aim_result$result)
  final_result = obtained_acc_G_mean(table0)
  Acc = final_result[[1]]
  Gmean = final_result[[2]]
  comp = data.frame(Acc, Gmean, the_number_select / samples_number * 100)
  names(comp) = c("Acc", "Gmean", "Mar")
  print(comp)
  saver = read.table("D:/program/data.csv", header = TRUE, sep = ",")
  saver = rbind(saver, comp)
  write.csv(saver, "D:/program/data.csv", row.names = FALSE)
}

for (number in 1:10) {
  model(1024, 1000, 5, "D:/program/pageblocks_train.csv", 3000, "D:/program/pageblocks_test.csv")
}
