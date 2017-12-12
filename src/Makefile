all: paper-examples

distribution:
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/distribution/decay.png				--command histogram --nr 1 --nb 1 --exponential_decay 0.01
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/distribution/nodecay.png				--command histogram --nr 1 --nb 1
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/distribution/decay-diffreplace.png			--command histogram --nr 1 --nb 1 --exponential_decay 0.01
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/distribution/nodecay-diffreplace.png		--command histogram --nr 1 --nb 1
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/distribution/decay-redstart.png			--command histogram --nr 5 --nb 1 --exponential_decay 0.01
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/distribution/nodecay-redstart.png			--command histogram --nr 5 --nb 1
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/distribution/decay-diffreplace-redstart.png		--command histogram --nr 5 --nb 1 --exponential_decay 0.01
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/distribution/nodecay-diffreplace-redstart.png	--command histogram --nr 5 --nb 1
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/distribution/decay-blackstart.png			--command histogram --nr 1 --nb 5 --exponential_decay 0.01
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/distribution/nodecay-blackstart.png			--command histogram --nr 1 --nb 5
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/distribution/decay-diffreplace-blackstart.png	--command histogram --nr 1 --nb 5 --exponential_decay 0.01
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/distribution/nodecay-diffreplace-blackstart.png	--command histogram --nr 1 --nb 5

probplot:
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/probplot/decay.png					--command probplot --nr 1 --nb 1 --exponential_decay 0.01
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/probplot/nodecay.png				--command probplot --nr 1 --nb 1
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/probplot/decay-diffreplace.png			--command probplot --nr 1 --nb 1 --exponential_decay 0.01
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/probplot/nodecay-diffreplace.png			--command probplot --nr 1 --nb 1
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/probplot/decay-redstart.png				--command probplot --nr 5 --nb 1 --exponential_decay 0.01
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/probplot/nodecay-redstart.png			--command probplot --nr 5 --nb 1
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/probplot/decay-diffreplace-redstart.png		--command probplot --nr 5 --nb 1 --exponential_decay 0.01
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/probplot/nodecay-diffreplace-redstart.png		--command probplot --nr 5 --nb 1
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/probplot/decay-blackstart.png			--command probplot --nr 1 --nb 5 --exponential_decay 0.01
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/probplot/nodecay-blackstart.png			--command probplot --nr 1 --nb 5
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/probplot/decay-diffreplace-blackstart.png		--command probplot --nr 1 --nb 5 --exponential_decay 0.01
	src/main.py --a 2 --b 0 --c 0 --d 1 --output output/probplot/nodecay-diffreplace-blackstart.png		--command probplot --nr 1 --nb 5

good:
	src/main.py --a 0.1 --b 0 --c 0 --d 0.1 --output output/good/decay.png					--command probplot --nr 1 --nb 1 --linear_surprise --exponential_decay 0.01
	src/main.py --a 0.1 --b 0 --c 0 --d 0.1 --output output/good/nodecay.png				--command probplot --nr 1 --nb 1 --linear_surprise 
	src/main.py --a 0.2 --b 0 --c 0 --d 0.1 --output output/good/decay-diffreplace.png			--command probplot --nr 1 --nb 1 --linear_surprise --exponential_decay 0.01
	src/main.py --a 0.2 --b 0 --c 0 --d 0.1 --output output/good/nodecay-diffreplace.png			--command probplot --nr 1 --nb 1 --linear_surprise 
	src/main.py --a 0.1 --b 0 --c 0 --d 0.1 --output output/good/decay-redstart.png				--command probplot --nr 5 --nb 1 --linear_surprise --exponential_decay 0.01
	src/main.py --a 0.1 --b 0 --c 0 --d 0.1 --output output/good/nodecay-redstart.png			--command probplot --nr 5 --nb 1 --linear_surprise 
	src/main.py --a 0.2 --b 0 --c 0 --d 0.1 --output output/good/decay-diffreplace-redstart.png		--command probplot --nr 5 --nb 1 --linear_surprise --exponential_decay 0.01
	src/main.py --a 0.2 --b 0 --c 0 --d 0.1 --output output/good/nodecay-diffreplace-redstart.png		--command probplot --nr 5 --nb 1 --linear_surprise 
	src/main.py --a 0.1 --b 0 --c 0 --d 0.1 --output output/good/decay-blackstart.png			--command probplot --nr 1 --nb 5 --linear_surprise --exponential_decay 0.01
	src/main.py --a 0.1 --b 0 --c 0 --d 0.1 --output output/good/nodecay-blackstart.png			--command probplot --nr 1 --nb 5 --linear_surprise 
	src/main.py --a 0.2 --b 0 --c 0 --d 0.1 --output output/good/decay-diffreplace-blackstart.png		--command probplot --nr 1 --nb 5 --linear_surprise --exponential_decay 0.01
	src/main.py --a 0.2 --b 0 --c 0 --d 0.1 --output output/good/nodecay-diffreplace-blackstart.png		--command probplot --nr 1 --nb 5 --linear_surprise

simple:
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/polya-urn-decay.png						--command singlerun --nr 1 --nb 1 --exponential_decay 0.01
	src/main.py --a 1 --b 0 --c 0 --d 1 --output output/polya-urn-nodecay.png					--command singlerun --nr 1 --nb 1

# oakland-examples-new:
# 	src/main.py --a 3.68511  --b 0 --c 0 --d 2.816466  --command probplot --nr 609  --nb 379  --ndraws 1000 --output output/paper-examples/oakland-top-2-1000-days-strong-prior.png
# 	src/main.py --a 3.68511  --b 0 --c 0 --d 2.816466  --command probplot --nr 6.09 --nb 3.79 --ndraws 1000 --output output/paper-examples/oakland-top-2-1000-days-weak-prior.png
# 	src/main.py --a 0.368511 --b 0 --c 0 --d 0.2816466 --command probplot --nr 609  --nb 379  --ndraws 1000 --linear_surprise    --output output/paper-examples/oakland-good-top-2-1000-days-strong-prior.png
# 	src/main.py --a 0.368511 --b 0 --c 0 --d 0.2816466 --command probplot --nr 6.09 --nb 3.79 --ndraws 1000 --linear_surprise    --output output/paper-examples/oakland-good-top-2-1000-days-weak-prior.png
# 	src/main.py --a 3.68511  --b 0 --c 0 --d 2.816466  --command probplot --nr 609  --nb 379  --ndraws 1000 --nruns 1000 --poisson --output output/paper-examples/oakland-top-2-poisson-1000-days-strong-prior.png
# 	src/main.py --a 3.68511  --b 0 --c 0 --d 2.816466  --command probplot --nr 6    --nb 3    --ndraws 1000 --nruns 1000 --poisson --output output/paper-examples/oakland-top-2-poisson-1000-days-weak-prior.png

oakland-examples:
	src/main.py --a 3.68511  --b 0 --c 0 --d 2.816466  --command probplot --nr 609  --nb 379  --ndraws 1000 --output output/paper-examples/oakland-top-2-1000-days-strong-prior.png
	src/main.py --a 3.68511  --b 0 --c 0 --d 2.816466  --command probplot --nr 6.09 --nb 3.79 --ndraws 1000 --output output/paper-examples/oakland-top-2-1000-days-weak-prior.png
	src/main.py --a 0.368511 --b 0 --c 0 --d 0.2816466 --command probplot --nr 609  --nb 379  --ndraws 1000 --linear_surprise    --output output/paper-examples/oakland-good-top-2-1000-days-strong-prior.png
	src/main.py --a 0.368511 --b 0 --c 0 --d 0.2816466 --command probplot --nr 6.09 --nb 3.79 --ndraws 1000 --linear_surprise    --output output/paper-examples/oakland-good-top-2-1000-days-weak-prior.png
	src/main.py --a 3.68511  --b 0 --c 0 --d 2.816466  --command probplot --nr 609  --nb 379  --ndraws 1000 --nruns 1000 --poisson --output output/paper-examples/oakland-top-2-poisson-1000-days-strong-prior.png
	src/main.py --a 3.68511  --b 0 --c 0 --d 2.816466  --command probplot --nr 6    --nb 3    --ndraws 1000 --nruns 1000 --poisson --output output/paper-examples/oakland-top-2-poisson-1000-days-weak-prior.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 609  --nb 7    --ndraws 1000 --output output/paper-examples/oakland-top-vs-random-1000-days-strong-prior.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 6.09 --nb 0.07 --ndraws 1000 --output output/paper-examples/oakland-top-vs-random-1000-days-weak-prior.png
	src/main.py --a 0.368511 --b 0 --c 0 --d 0.235592 --command probplot --nr 609  --nb 7    --ndraws 1000 --linear_surprise    --output output/paper-examples/oakland-good-top-vs-random-1000-days-strong-prior.png
	src/main.py --a 0.368511 --b 0 --c 0 --d 0.235592 --command probplot --nr 6.09 --nb 1  --ndraws 1000 --linear_surprise    --output output/paper-examples/oakland-good-top-vs-random-1000-days-weak-prior.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 609  --nb 7    --ndraws 1000 --poisson --output output/paper-examples/oakland-top-vs-random-poisson-1000-days-strong-prior.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 6    --nb 1    --ndraws 1000 --poisson --output output/paper-examples/oakland-top-vs-random-poisson-1000-days-weak-prior.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.816466 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-nodecay.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.816466 --nr 6.09 --nb 3.79 --ndraws 1000 --command probplot --nruns 1000 --output output/paper-examples/oakland-top-2-1000-days-weak-prior-poisson-binomial-nodecay.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.816466 --poisson --linear_surprise --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-nodecay-with-surprise.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.816466 --poisson --linear_surprise --nr 6.09 --nb 3.79 --ndraws 1000 --command probplot --nruns 1000 --output output/paper-examples/oakland-top-2-1000-days-weak-prior-poisson-binomial-nodecay-with-surprise.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 609  --nb 7    --ndraws 1000 --nruns 1000 --poisson --output output/paper-examples/oakland-top-vs-random-1000-days-strong-prior-poisson-binomial-nodecay.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 6    --nb 1    --ndraws 1000 --nruns 1000 --poisson --output output/paper-examples/oakland-top-vs-random-1000-days-weak-prior-poisson-binomial-nodecay.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 609  --nb 7    --ndraws 1000 --nruns 1000 --poisson --linear_surprise --output output/paper-examples/oakland-top-vs-random-1000-days-strong-prior-poisson-binomial-nodecay-with-surprise.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 6    --nb 1    --ndraws 1000 --nruns 1000 --poisson --linear_surprise --output output/paper-examples/oakland-top-vs-random-1000-days-weak-prior-poisson-binomial-nodecay-with-surprise.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.816466 --poisson --nr 6.09 --nb 3.79 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-weak-prior-poisson-binomial-decay.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.816466 --poisson --linear_surprise --nr 6.09 --nb 3.79 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-weak-prior-poisson-binomial-decay-with-surprise.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 6    --nb 1    --ndraws 1000 --nruns 1000 --poisson --exponential_decay 0.01 --output output/paper-examples/oakland-top-vs-random-1000-days-weak-prior-poisson-binomial-decay.png
	src/main.py --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 6    --nb 1    --ndraws 1000 --nruns 1000 --poisson --linear_surprise --exponential_decay 0.01 --output output/paper-examples/oakland-top-vs-random-1000-days-weak-prior-poisson-binomial-decay-with-surprise.png

paper-examples:
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0 --c 0 --d 2.816466 --poisson --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0 --c 0 --d 2.816466 --poisson --linear_surprise --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-with-surprise.png
	src/main.py --lambdaa 3.68511 --lambdab 2.35592 --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 609  --nb 7    --ndraws 1000 --nruns 1000 --poisson --exponential_decay 0.01 --output output/paper-examples/oakland-top-vs-random-1000-days-strong-prior-poisson-binomial-decay.png
	src/main.py --lambdaa 3.68511 --lambdab 2.35592 --a 3.68511 --b 0 --c 0 --d 2.35592 --command probplot --nr 609  --nb 7    --ndraws 1000 --nruns 1000 --poisson --linear_surprise --exponential_decay 0.01 --output output/paper-examples/oakland-top-vs-random-1000-days-strong-prior-poisson-binomial-decay-with-surprise.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 7.37022 --b 2.816466 --c 3.68511 --d 5.632932 --poisson --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 7.37022 --b 2.816466 --c 3.68511 --d 5.632932 --poisson --linear_surprise --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-surprise.png
	src/main.py --lambdaa 3.68511 --lambdab 2.35592 --a 7.37022 --b 2.35592 --c 3.68511 --d 4.71184 --command probplot --nr 609  --nb 7    --ndraws 1000 --nruns 1000 --poisson --exponential_decay 0.01 --output output/paper-examples/oakland-top-vs-random-1000-days-strong-prior-poisson-binomial-decay-mixed.png
	src/main.py --lambdaa 3.68511 --lambdab 2.35592 --a 7.37022 --b 2.35592 --c 3.68511 --d 4.71184 --command probplot --nr 609  --nb 7    --ndraws 1000 --nruns 1000 --poisson --linear_surprise --exponential_decay 0.01 --output output/paper-examples/oakland-top-vs-random-1000-days-strong-prior-poisson-binomial-decay-mixed-with-surprise.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 7.37022 --b 2.816466 --c 3.68511 --d 5.632932 --poisson --weighted_surprise 3.68511 2.816466 0.5 0.5 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-weightedsurprise.png
### varying wd with NO fix
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0 --c 0 --d 2.816466 --poisson --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed_wd1.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0.2816466 --c 0.368511 --d 2.816466 --poisson --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed_wd0.9.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0.5632932 --c 0.737022 --d 2.816466 --poisson --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed_wd0.8.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0.8449398 --c 1.10553 --d 2.816466 --poisson --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed_wd0.7.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 1.1265864 --c 1.474044 --d 2.816466 --poisson --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed_wd0.6.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 1.408233 --c 1.842555 --d 2.816466 --poisson --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed_wd0.5.png
### varying wd with the fix
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0 --c 0 --d 2.816466 --poisson --weighted_surprise 3.68511 2.816466 1 0 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-weightedsurprise_wd1.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0.2816466 --c 0.368511 --d 2.816466 --poisson --weighted_surprise 0.368511 0.2816466 0.9 0.1 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-weightedsurprise_wd0.9.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0.5632932 --c 0.737022 --d 2.816466 --poisson --weighted_surprise 0.737022 0.5632932 0.8 0.2 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-weightedsurprise_wd0.8.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0.8449398 --c 1.10553 --d 2.816466 --poisson --weighted_surprise 1.10553 0.8449398 0.7 0.3 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-weightedsurprise_wd0.7.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 1.1265864 --c 1.474044 --d 2.816466 --poisson --weighted_surprise 1.474044 1.1265864 0.6 0.4 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-weightedsurprise_wd0.6.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 1.408233 --c 1.842555 --d 2.816466 --poisson --weighted_surprise 1.842555 1.408233 0.5 0.5 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-weightedsurprise_wd0.5.png
#### varying wd with partial fix
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0 --c 0 --d 2.816466 --poisson --partial_surprise 3.68511 2.816466 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-partialsurprise_wd1.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0.2816466 --c 0.368511 --d 2.816466 --poisson --partial_surprise 0.368511 0.2816466 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-partialsurprise_wd0.9.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0.5632932 --c 0.737022 --d 2.816466 --poisson --partial_surprise 0.737022 0.5632932 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-partialsurprise_wd0.8.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 0.8449398 --c 1.10553 --d 2.816466 --poisson --partial_surprise 1.10553 0.8449398 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-partialsurprise_wd0.7.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 1.1265864 --c 1.474044 --d 2.816466 --poisson --partial_surprise 1.474044 1.1265864 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-partialsurprise_wd0.6.png
	src/main.py --lambdaa 3.68511 --lambdab 2.816466 --a 3.68511 --b 1.408233 --c 1.842555 --d 2.816466 --poisson --partial_surprise 1.842555 1.408233 --nr 609 --nb 379 --ndraws 1000 --command probplot --nruns 1000 --exponential_decay 0.01 --output output/paper-examples/oakland-top-2-1000-days-strong-prior-poisson-binomial-decay-mixed-with-partialsurprise_wd0.5.png
##	src/main.py --lambdaa 3.68511 --lambdab 2.35592 --a 7.37022 --b 2.35592 --c 3.68511 --d 4.71184 --command probplot --nr 609  --nb 7    --ndraws 1000 --nruns 1000 --poisson --weighted_surprise 3.68511 2.35592 0.5 0.5 --exponential_decay 0.01 --output output/paper-examples/oakland-top-vs-random-1000-days-strong-prior-poisson-binomial-decay-mixed-with-weightedsurprise.png

clean:
	rm output/paper-examples/*
	rm output/good/*
	rm output/probplot/*
	rm output/distribution/*


