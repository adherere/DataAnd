# load package
library(readr)
library(factoextra)  
library(ggplot2)  




#--------------------------------------The data is for ITV calculation values after PCA dimensionality reduction.------------------------------------------
file_path <- "inputTraits.csv"
Origin_trait_data <- read_csv(file_path)


phenotype_traits <- Origin_trait_data[, c("Angle", "Height", "Stem", "Chl", "LMA", "LMDC", "SSD", "SMC", "N", "P")]

phenotype_traits_clean <- na.omit(phenotype_traits)

scaled_features <- scale(phenotype_traits_clean)

# Perform PCA analysis and extract principal components.
pca_result <- prcomp(scaled_features, center = TRUE, scale. = TRUE)
pca_scores_top3 <- pca_result$x[, 1:3]

pca_scores_df <- as.data.frame(pca_scores_top3)
colnames(pca_scores_df) <- c("PC1", "PC2", "PC3")
extra_attributes <- Origin_trait_data[, c("pdname", "Species", "Competition_treatment")]

# Calculate ITV based on PCA
trait_data <- cbind(extra_attributes, pca_scores_df)
trait_names <- colnames(trait_data)[4:6]  
species <- colnames(trait_data)[2]

trait_data$group <- paste(trait_data$Species, trait_data$Competition_treatment, sep = "_")

Hypervolume_value <- function(ni, trait_data, trait, sample_size, quantile) {
  if (length(ni) > 2) {
    data <- trait_data[ni, ]
    if (sample_size == "all") {
      sample_data <- data[, colnames(data) %in% trait]
    } else {
      ind_num <- nrow(data)
      if (sample_size > ind_num) {
        stop("Sample size cannot be greater than the number of samples.")
      }
      sample_num <- sample(1:ind_num, sample_size)
      sample_data <- data[sample_num, colnames(data) %in% trait]
    }
    
    # Calculate bandwidth.
    bandwidth <- estimate_bandwidth(data = sample_data,method="silverman-1d")
    # Calculate hypervolume
    hypervolume <- hypervolume_gaussian(data = sample_data, name = unique(sample_data$pdname), kde.bandwidth = bandwidth, quantile.requested = quantile)
    hv <- hypervolume@Volume
  } else {
    hv <- NA
  }
  return(hv)
}

# Function to simulate hypervolume values across different species
Sim_Hypervolume <- function(n, trait_data, trait, sample_size, quantile, Hypervolume_value) {
  hvs <- tapply(1:nrow(trait_data), trait_data$group, Hypervolume_value, trait_data, trait, sample_size, quantile)
  return(hvs)
}


trait_data[, trait_names] <- scale(trait_data[, trait_names])

# Calculate ITV
ni <- 1:nrow(trait_data)  
sample_size <- "all"  
quantile <- 0.95 
Nrep <- 1
hv <- lapply(1:Nrep, Sim_Hypervolume, trait_data, trait_names, sample_size, quantile, Hypervolume_value)

output_file_path <- "output_path"
save(hv, file = paste0(output_file_path, "/HV_year2All2.RData"))


