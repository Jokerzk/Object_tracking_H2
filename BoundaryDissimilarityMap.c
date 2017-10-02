#include"BoundaryDissimilarityMap.h"

cv::Mat getBoundaryDissimilarityMap(cv::Mat Src, int boundarysize)
{
	int width = Src.cols;
	int height = Src.rows;
	//int size = width*height;

	cv::Mat Labimg = Mat(Src.rows, Src.cols, CV_32FC3, Scalar(0));
	cv::cvtColor(Src, Labimg, CV_BGR2Lab);

	Group CurrentImg;
	KMeans(Labimg, width, height, boundarysize, CurrentImg.clusters, CurrentImg.means);
	getCovmatrixInvert(width, height, CurrentImg.clusters, CurrentImg.covinvert, CurrentImg.vec_binv);//此处得到协方差矩阵的逆；

	/*计算马氏距离*/
	float* MahalMap = (float *)malloc(height * width * sizeof(float));
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j += 1)
		{
			int total_num = 0;
			float total_distance = 0;
			//float max_dis = 0;
			for (int g = 0; g < k; g++)
			{
				if (!CurrentImg.vec_binv[g])
				{
					continue;
				}
				float submatrix[3];
				submatrix[0] = Labimg.data[(i*width + j)*3 + 0] - CurrentImg.means[g].attrL;
				submatrix[1] = Labimg.data[(i*width + j) * 3 + 1] - CurrentImg.means[g].attra;
				submatrix[2] = Labimg.data[(i*width + j) * 3 + 2] - CurrentImg.means[g].attrb;

				float S = sqrt(submatrix[0] * (submatrix[0] * CurrentImg.covinvert[g][0][0] + submatrix[1] * CurrentImg.covinvert[g][1][0] + submatrix[2] * CurrentImg.covinvert[g][2][0]) +
							   submatrix[1] * (submatrix[0] * CurrentImg.covinvert[g][0][1] + submatrix[1] * CurrentImg.covinvert[g][1][1] + submatrix[2] * CurrentImg.covinvert[g][2][1]) +
							   submatrix[2] * (submatrix[0] * CurrentImg.covinvert[g][0][2] + submatrix[1] * CurrentImg.covinvert[g][1][2] + submatrix[2] * CurrentImg.covinvert[g][2][2]));

				//if (max_dis < S)
				//{
				//	max_dis = S;
				//}
				total_distance += CurrentImg.clusters[g].size() * S;
				total_num += CurrentImg.clusters[g].size();
			}
			//MahalMap[i*width + j] = (CurrentImg.clusters[0].size()*S[0] + CurrentImg.clusters[1].size()*S[1] + CurrentImg.clusters[2].size()*S[2]) / (CurrentImg.clusters[0].size() + CurrentImg.clusters[1].size() + CurrentImg.clusters[2].size());
			MahalMap[i*width + j] = total_distance / total_num;
			//MahalMap[i*width + j] = max_dis;
			//cout << MahalMap[i*width + j] << endl;
		}
	}
	cv::Mat hann1t = Mat(cv::Size(width,1),CV_32FC1);
	cv::Mat hann2t = Mat(cv::Size(1,height),CV_32FC1);
	cv::Mat hann_2D = Mat(cv::Size(width,height),CV_32FC1,cv::Scalar(0));
	for (int i = 0; i < width; ++i)
	{
		hann1t.at<float>(0,i) = 0.5*(1-cos(2*pi*i/(width-1)));
	}
	for (int i = 0; i < height; ++i)
	{
		hann2t.at<float>(i,0) = 0.5*(1-cos(2*pi*i/(height-1)));
	}
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			hann_2D.at<float>(i,j) = hann2t.at<float>(i,0)*hann1t.at<float>(0,j);
		}
	}
	cv::Mat salient_map = Mat(height,width,CV_32FC1,MahalMap);
	salient_map = salient_map.mul(hann_2D);
	cv::normalize(salient_map,salient_map,255,0,CV_MINMAX);
	salient_map.convertTo(salient_map,CV_8UC1);

	return salient_map;
}
 
//获得当前簇集的协方差逆矩阵
void getCovmatrixInvert(int width, int height, vector<Tuple> clusters[k], float covinvert[k][3][3], vector<bool> &vec_binv)
{
	for (int  j = 0; j < k; j++)
	{
		int num = clusters[j].size();
		vector <Mat> data(k), covar(k), mean(k);
		data[j] = Mat(num, 3, CV_32FC1);
		//double sumLL = 0, sumLa = 0, sumLb = 0, sumaL = 0, sumaa = 0, sumab = 0, sumbL = 0, sumba = 0, sumbb = 0;
		for (int i = 0; i < num; i++)
		{
			data[j].at<float>(i, 0) = clusters[j][i].attrL;
			data[j].at<float>(i, 1) = clusters[j][i].attra;
			data[j].at<float>(i, 2) = clusters[j][i].attrb;
		}
		if (num == 0)
		{
			vec_binv.push_back(false);
			continue;
		}
		calcCovarMatrix(data[j], covar[j], mean[j], CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);
		CvMat cvcovar= covar[j];
		if (cvDet(&cvcovar) == 0)//判断是否存在矩阵行列式为0，若存在则由马氏距离所求距离全部为0（矩阵奇异）；
		{
			vec_binv.push_back(false);
			continue;
		}
		else
		{
			vec_binv.push_back(true);
		}
		CvMat *covarinvert = cvCreateMat(3, 3, CV_32FC1);
		cvInvert(&cvcovar, covarinvert);
		for (int row = 0; row<covarinvert->height; row++)
		{
			float* pData = (float*)(covarinvert->data.ptr + row*covarinvert->step);
			for (int col = 0; col<covarinvert->width; col++)
			{
				covinvert[j][row][col] = *pData;
				pData++;
			}
		}
	}
}

//计算两个元组在Lab空间内的欧几里得距离  
float getDistLab(Tuple t1, Tuple t2)
{
	return sqrt((t1.attrL - t2.attrL) * (t1.attrL - t2.attrL) + (t1.attra - t2.attra) * (t1.attra - t2.attra) + (t1.attrb - t2.attrb) * (t1.attrb - t2.attrb));
}

//根据质心，决定当前元组属于哪个簇  
int clusterOfTuple(Tuple means[], Tuple tuple)//tuple:当前元素
{
	float dist = getDistLab(means[0], tuple);
	float tmp;
	int label = 0;//标示属于哪一个簇  
	for (int i = 1; i<k; i++)
	{
		tmp = getDistLab(means[i], tuple);
		if (tmp<dist) 
		{ dist = tmp; label = i; }
	}
	return label;
}

//获得给定簇集的平方误差,用来确定收敛界限  
float getVar(vector<Tuple> clusters[], Tuple means[])
{
	float var = 0;
	for (int i = 0; i < k; i++)
	{
		vector<Tuple> t = clusters[i];
		for (size_t j = 0; j< t.size(); j++)
		{
			var += getDistLab(t[j], means[i]);
		}
	}
	return var;
}
//获得当前簇的均值（质心）
Tuple getMeans(vector<Tuple> cluster)
{

	int num = cluster.size();
	double meansX = 0, meansY = 0, meansZ = 0;
	Tuple t;
	for (int i = 0; i < num; i++)
	{
		meansX += cluster[i].attrL;
		meansY += cluster[i].attra;
		meansZ += cluster[i].attrb;
	}
	t.attrL = meansX / num;
	t.attra = meansY / num;
	t.attrb = meansZ / num;
	return t;
}
//获取当前图片的种子，选取左上左下右上右下四个像素作为聚类种子
vector<Tuple> getSeeds(cv::Mat imgLab, int width, int height, vector<Tuple> Seeds)
{
	int size = width * height;
	Seeds[0].attrL = imgLab.data[3 * int((width - 1) * 0.5) + 0];
	Seeds[0].attra = imgLab.data[3 * int((width - 1) * 0.5) + 1];
	Seeds[0].attrb = imgLab.data[3 * int((width - 1) * 0.5) + 2];
	Seeds[1].attrL = imgLab.data[3 * width * int((height - 1) * 0.5) + 0];
	Seeds[1].attra = imgLab.data[3 * width * int((height - 1) * 0.5) + 1];
	Seeds[1].attrb = imgLab.data[3 * width * int((height - 1) * 0.5) + 2];
	Seeds[2].attrL = imgLab.data[3 * (size - 1) + 0];
	Seeds[2].attra = imgLab.data[3 * (size - 1) + 1];
	Seeds[2].attrb = imgLab.data[3 * (size - 1) + 2];
	return Seeds;
}
vector<Tuple>* KMeans(cv::Mat imgLab, int width, int height, int boundarysize, vector<Tuple> clusters[k], Tuple means[k])
{
	int datanum = width * height - (width - boundarysize*2)*(height - boundarysize*2);
	vector<Tuple> tuples;
	for (int i = 0; i < height; i++)
	{
		Tuple temp;
		if (i < boundarysize || i >= (height - boundarysize))
		{
			for (int j = 0; j < width; j++)
			{
				temp.attrL = imgLab.data[(i*width + j) * 3 + 0];
				temp.attra = imgLab.data[(i*width + j) * 3 + 1];
				temp.attrb = imgLab.data[(i*width + j) * 3 + 2];
				tuples.push_back(temp);
			}
		}
		else
		{
			for (int j = 0; j < width; j++)
			{
				if (j<boundarysize || j >= (width - boundarysize))
				{
					temp.attrL = imgLab.data[(i*width + j) * 3 + 0];
					temp.attra = imgLab.data[(i*width + j) * 3 + 1];
					temp.attrb = imgLab.data[(i*width + j) * 3 + 2];
					tuples.push_back(temp);
				}	
			}
		}
	}
	/*此处预先指定四个边缘点为3个簇的质心*/
	vector<Tuple> seeds(k);
	seeds = getSeeds(imgLab, width, height, seeds);
	for (int i = 0; i < k; i++)
	{
		means[i].attrL = seeds[i].attrL;
		means[i].attra = seeds[i].attra;
		means[i].attrb = seeds[i].attrb;
	}
	int lable = 0;
	/*根据默认的质心给簇赋值*/ 
	for (int i = 0; i != datanum; ++i)
	{
		lable = clusterOfTuple(means, tuples[i]);
		clusters[lable].push_back(tuples[i]);
	}
	float oldVar = -1;
	float newVar = getVar(clusters, means);
	/*当新旧函数值相差不到1即准则函数值不发生明显变化时，算法终止*/
	while (abs(newVar - oldVar) >= 10) 
	{
		for (int i = 0; i < k; i++) 
		{
			means[i] = getMeans(clusters[i]); 
		}
		oldVar = newVar;
		newVar = getVar(clusters, means);
		for (int i = 0; i < k; i++)   
		{
			clusters[i].clear();
		}
		/*根据新的质心获得新的簇*/
		for (size_t i = 0; i != tuples.size(); ++i)
		{
			lable = clusterOfTuple(means, tuples[i]);
			clusters[lable].push_back(tuples[i]);
		}
	}
	return clusters;
}
