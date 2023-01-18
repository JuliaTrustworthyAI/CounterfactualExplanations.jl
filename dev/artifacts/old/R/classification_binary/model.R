run_model <- function(artifact_path,root="dev/artifacts/R/classification_binary") {
  library(torch)
  set.seed(1)
  X <- torch_load(file.path(root,"X.pt"))
  ys <- torch_load(file.path(root,"ys.pt"))

  # Model:
  mlp <- nn_module(
    initialize = function() {
      self$layer1 <- nn_linear(2, 32)
      self$layer2 <- nn_linear(32, 1)
    },
    forward = function(input) {
      input <- self$layer1(input)
      input <- nnf_sigmoid(input)
      input <- self$layer2(input)
      input
    }
  )
  model <- mlp()
  optimizer <- optim_adam(model$parameters, lr = 0.1)
  loss_fun <- nnf_binary_cross_entropy_with_logits
  for (epoch in 1:100) {
    model$train()  
    
    # Compute prediction and loss:
    output <- model(X)[,1]
    loss <- loss_fun(output, ys)
    
    # Backpropagation:
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()
    
    cat(sprintf("Loss at epoch %d: %7f\n", epoch, loss$item()))
  }

  save_dir <- file.path(artifact_path,"classification_binary")
  dir.create(save_dir)
  torch_save(model, file.path(save_dir,"model.pt"))
}
