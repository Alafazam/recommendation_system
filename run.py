from app import recommeder

V,M,U,O,M_genere = loadMovieLens()

userid = random.randrange(0, 943)

getUser(userid, True)

overall,M = rated_rankings()

kkk = k_similar_users(V, userid, k=10)

SU = get_similar_users(552)

final_reccom = getRecommendations(V, userid, SU)
print "Movies recommendations for current user, based on his prefrences "
print "\n".join(final_reccom)
