import boto3, csv, io, json, re, os, sys, pprint, time, random
from time import gmtime, strftime
from botocore.client import Config
import numpy as np
from scipy.sparse import lil_matrix

bucket = sys.argv[1]
glue_database = sys.argv[2]

current_timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())

#Function for executing athena queries
def run_query(query, database, s3_output):
    client = boto3.client('athena')
    print("Running Athena Query: %s" % (query))
    response = client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database
            },
        ResultConfiguration={
            'OutputLocation': s3_output,
            }
        )
    execution_id = response['QueryExecutionId']
    print('Execution ID: ' + execution_id)

    while True:
        stats = client.get_query_execution(
            QueryExecutionId=execution_id
        )
        status = stats['QueryExecution']['Status']['State']
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(0.2)  # 200ms

    if status == 'SUCCEEDED':
        results = client.get_query_results(
            QueryExecutionId=execution_id
        )

    if status == 'FAILED':
        print(stats['QueryExecution']['Status']['StateChangeReason'])

    return results

table = "u_data"
s3_athena_ouput = "s3://%s/athena-results/" % (bucket)
query = 'SELECT count(distinct userid) AS userCnt, count(distinct movieid) AS movieCnt FROM "%s"."%s";' % (glue_database, table)
res = run_query(query, glue_database, s3_athena_ouput)

nbUsers=int(res["ResultSet"]["Rows"][1]["Data"][0]["VarCharValue"])
nbMovies=int(res["ResultSet"]["Rows"][1]["Data"][1]["VarCharValue"])
nbFeatures=nbUsers+nbMovies
print("Table u_data has %s users and %s movies" % (nbUsers, nbMovies))

s3 = boto3.resource('s3')
s3.Bucket(bucket).download_file('movielens-data/u.data/data.csv', 'u.data')

# Pick 10 ratings per user and save as test data set ua.test
# Save the rest as training data set ua.base
testRatingsByUser = {}
maxRatingsByUser = 10
for userId in range(nbUsers):
    testRatingsByUser[str(userId)]=0

with open('u.data','r') as f, open('ua.base','w') as uabase, open('ua.test','w') as uatest:
    filedata=csv.reader(f,delimiter='\t')
    next(filedata, None) # skip headers
    uabasewriter = csv.writer(uabase, delimiter='\t')
    uatestwriter = csv.writer(uatest, delimiter='\t')
    nbRatingsTrain=0
    nbRatingsTest=0
    for userId,movieId,rating,timestamp in filedata:
        if testRatingsByUser[str(int(userId)-1)] < maxRatingsByUser:
            uatestwriter.writerow([userId,movieId,rating,timestamp])
            testRatingsByUser[str(int(userId)-1)] = testRatingsByUser[str(int(userId)-1)] + 1
            nbRatingsTest=nbRatingsTest+1
        else:
            uabasewriter.writerow([userId,movieId,rating,timestamp])
            nbRatingsTrain=nbRatingsTrain+1

# shuffle the training data to allow for minibatching
with open('ua.base','r') as uabase, open('ua.base.shuffled','w') as uabase_shuffled:
    uabase_lines = uabase.readlines()
    random.shuffle(uabase_lines)
    uabase_shuffled.writelines(uabase_lines)

prepdata_result = {
  "Parameters":
    {
        "TrainingData": "ua.base.shuffled",
        "TestData": "ua.test",
        "NbUsers": nbUsers,
        "NbMovies": nbMovies,
        "NbFeatures": nbFeatures,
        "NbRatingsTrain": nbRatingsTrain,
        "NbRatingsTest": nbRatingsTest,
        "Timestamp": current_timestamp
    }
}

pprint.pprint(prepdata_result)

json_prepdata_result = json.dumps(prepdata_result)

with open('prepdata_result.json', 'w') as prepdata_result_file:
    prepdata_result_file.write(json_prepdata_result)
