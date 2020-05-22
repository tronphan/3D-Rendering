import pandas as pd 

if __name__ == "__main__":
	path = "dataset/blur.csv"
	variances = pd.read_csv(path)
	variances.hist()
	