Geolocation of social media users based on text and network information



This is the implementation of the geolocation models described in the following two papers:

@InProceedings{rahimi2015exploiting,

author="Rahimi, Afshin
and Vu, Duy
and Cohn, Trevor
and Baldwin, Timothy",

title="Exploiting Text and Network Context for Geolocation of Social Media Users",

booktitle="Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",

year="2015",

publisher="Association for Computational Linguistics",

pages="1362--1367",

location="Denver, Colorado",

url="http://aclweb.org/anthology/N15-1153"
}



@InProceedings{rahimi2015twitter,

author="Rahimi, Afshin
and Cohn, Trevor
and Baldwin, Timothy",

title="Twitter User Geolocation Using a Unified Text and Network Prediction Model",

booktitle="Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",

year="2015",

publisher="Association for Computational Linguistics",

pages="630--636",

location="Beijing, China",

url="http://aclweb.org/anthology/P15-2104"
}


The models include text-based classification, network-based label propagation (regression)
and network-based label propagation (classification).

For the classification models, the real-valued coordinates of the training points are clustered using
k-d tree and each cluster (region) is assigned a label. This implementation expects the clusters to be
written into a text file where training point members of each cluster are written in one line tab-separated.
For example int the following example we have 2 clusters (regions) each with two training points:


    lat1,lon1	lat2,lon2
    lat3,lon3	lat4,lon4

This program expects three separate training, dev and test gzipped files in each, a user is represented as a line with the following
format:

    username	latitude	longitude	aggregated-tweets-of-user-including-mentions

    
Note: The datasets used are different from those used in:
Wing, Benjamin, and Jason Baldridge. "Hierarchical Discriminative Classification for Text-Based Geolocation." EMNLP. 2014.
in that we have rebuilt the datasets to include mention information.