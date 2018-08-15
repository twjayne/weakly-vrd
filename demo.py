import run.shared as shared

runner = shared.Runner()
runner.setup()

if __name__ == '__main__':
	runner.train()
