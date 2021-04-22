# socialSig-mig

## Current Results






### Weighted Trasnfer Learning Models


| Transfer |  Loss Function	|     MAE	  |      r2       |	 Quantity Error	 |  Allocation Error  |
|----------|----------------|-------------|---------------|------------------|--------------------|
|   Yes	   |     Normal	    | 193.8811298 |	0.1925204041  |	  190282.6378	 |    261654.2757     |
|   No	   |     Normal	    | 200.3015758 |	0.1995049326  |	  351327.1716	 |    115575.8016     |
|   No	   |    Weighted	| 94.65266588 |	0.4373283115  |	  129829.9505	 |    90805.41368     |
|   Yes	   |    Weighted	| 107.8059514 |	0.4457912232  |	  196465.777	 |    54829.89573     |



### Definitions & Weights

**Transfer** <br/>
Yes = Used pre-trained weights from within Mexico model<br/>
No = Used only pre-trained ImageNet weights


**Loss Function** <br/>
Normal = L1 Loss (Sum of absolute errors) <br/>
Weighted = Weighted L1 Loss (Sum of absolute errors multipled by the weight of their class)
		
		
		
		
**Weights**  
*Number of migrants:*  |     Number      |    Weight (1/n)   |
|----------------------|-----------------|-------------------|
|          0	       |     1/185	     |   0.005405405405  |
|       (0, 14)	       |     1/384	     |   0.002604166667  |
|      (14, 198)	   |     1/1177	     |   0.000849617672  |
|     (198, 600)	   |     1/401	     |   0.002493765586  |
|    (600, 34582)      |	 1/183	     |   0.005464480874  |






**Outdated**  

|       Model	    |       MAE	     |          R2	     |  Quantity Error	|  Allocation Error  |
|-------------------|----------------|-------------------|------------------|--------------------|
| Decision Tree	    |              	 |              	 |                  |                    |
| KNN	            |                |              	 |          	    |                    |
| Random Forest	    |              	 |              	 |              	|                    |
| Neural Network	|              	 |              	 |              	|                    |
| socialSigNoDrop	|  3.854131323	 |  0.00007840973309 |    17167.04453	|     59915.58194    |
| socialSign - VB	|          	     |          	     |          	    |                    |