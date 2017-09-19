{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Hard Drive Failure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Objective:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Explore hard drive failure data and provide device failure analysis using the Kaplan-Meier model.  "
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    "Data:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is data provided by BackBlaze.  They are an online file backup and storage company. They have posted their log files of hard drives.  There are 256 S.M.A.R.T statistics recorded for each drive.  "
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    "Data Prep:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Because this data is large, 3.5gb for 2 years.  I have ran the first part on my system and uploaded a smaller simpler table of the values that are important for the model.  The below script has taken the 3.5gb and made it a 5.2mb file that can be followed using ipython.\n",
    "\n"
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    "Import dependencies:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sb\n",
    "import lifelines as sa\n",
    "import glob, os\n",
    "from collections import OrderedDict\n",
    "from lifelines.utils import concordance_index, k_fold_cross_validation\n",
    "import patsy as pt\n",
    "from datetime import datetime\n",
    "from pylab import rcParams\n",
    "from sklearn.preprocessing import LabelEncoder, Imputer\n",
    "#from sklearn.cross_validation import train_test_split\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import MinMaxScaler, StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.base import clone\n",
    "from itertools import combinations\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.pipeline import Pipeline"
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    "Do not run, as the files are too large.  In place to show work:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.chdir(\"C:\\\\Users\\\\Andrew\\\\Downloads\\\\hdd\\\\\")\n",
    "\n",
    "allFiles = glob.glob(\"*.csv\")\n",
    "df = pd.DataFrame()\n",
    "list_ = []\n",
    "for file_ in allFiles:\n",
    "    df = pd.read_csv(file_,index_col=None, header=0, usecols=[0, 1, 2, 3,4, 20])\n",
    "    list_.append(df)\n",
    "df = pd.concat(list_)\n",
    "df.reset_index(inplace=True)\n",
    "\n",
    "os.chdir(\"C:\\\\Users\\\\Andrew\\\\Documents\")\n",
    "\n",
    "df[\"mindate\"] = df[\"date\"].groupby(df[\"serial_number\"]).transform('min')\n",
    "df[\"maxdate\"] = df[\"date\"].groupby(df[\"serial_number\"]).transform('max')\n",
    "df[\"minhours\"] = df[\"smart_9_raw\"].groupby(df[\"serial_number\"]).transform('min')\n",
    "df[\"maxhours\"] = df[\"smart_9_raw\"].groupby(df[\"serial_number\"]).transform('max')\n",
    "df[\"nrec\"] = df[\"date\"].groupby(df[\"serial_number\"]).transform('count')\n",
    "\n",
    "df = df[[\"date\", \"serial_number\",\"model\",\"capacity_bytes\",\"mindate\",\"maxdate\",\n",
    "        \"minhours\", \"maxhours\",\"nrec\",\"failure\"]]\n",
    "\n",
    "df = df.sort_values(\"failure\",ascending=False)\n",
    "df = df.drop_duplicates([\"serial_number\"],keep=\"first\")\n",
    "df.reset_index(inplace=True)\n",
    "\n",
    "#Save off file\n",
    "df.to_csv(\"HDD-log.csv\", index=False)\n"
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    ""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    ""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    ""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"HDD-log.csv\")\n",
    "\n",
    "#check for anything odd with failed\n",
    "df[\"failure\"].value_counts()\n",
    "\n",
    "#check for any hard drive that is just too small\n",
    "df[\"capacity_bytes\"].value_counts()\n",
    "\n",
    "#adjust to TB size\n",
    "df[\"capacity\"] = df[\"capacity_bytes\"].apply(lambda x: (round(x/1000000000000,2)))\n",
    "df.groupby([\"capacity\"]).size()\n",
    "\n",
    "#drop hdd that are too small of records or too large\n",
    "df = df.loc[(df['capacity']<6.5) & (df['capacity']>1.5)]\n",
    "\n",
    "#Create the make/model of the drives\n",
    "df[\"make\"] = df[\"model\"].apply(lambda x:x.split()[0])\n",
    "df[\"model\"] = df[\"model\"].apply(lambda x: x.split()[1] if len(x.split())>1\n",
    "else x)\n",
    "df.groupby([\"make\",\"model\"]).size()\n",
    "\n",
    "#Create the Seagate make and Hitachi\n",
    "df[\"make\"] = df[\"make\"].apply(lambda x:\"Seagate\" if x[:2]== \"ST\" else x)\n",
    "df.loc[df[\"make\"] == \"HGST\",\"make\"] = \"Hitachi\"\n",
    "\n",
    "#Visual inspection of the data\n",
    "gp = df.groupby([\"make\",\"capacity\"]).size().unstack()\n",
    "sb.heatmap(gp, mask=pd.isnull(gp), robust=True, square=True,cbar=False)\n",
    "\n"
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    ""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Shrink to just Hitachi, Seagate, WD\n",
    "df = df.loc[df[\"make\"].isin([\"Hitachi\", \"Seagate\", \"WDC\"])]\n",
    "\n",
    "#visual inspection of date start and ending\n",
    "df[\"mindate\"] = pd.to_datetime(df[\"mindate\"])\n",
    "df[\"maxdate\"] = pd.to_datetime(df[\"maxdate\"])\n",
    "df['mindateym'] = df['mindate'].apply(lambda x: x.strftime('%Y%m'))\n",
    "df['maxdateym'] = df['maxdate'].apply(lambda x: x.strftime('%Y%m'))\n",
    "\n",
    "#Grouping by Make and Capacity\n",
    "gp = df.groupby([\"make\",\"capacity\"]).size().unstack()\n",
    "gp[pd.isnull(gp)]=0\n",
    "fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,4)\n",
    "                         , squeeze=False, sharex=True, sharey=True)\n",
    "sb.heatmap(gp, annot=True, fmt='.0f', ax=axes[0,0])\n",
    "\n",
    "gp = df.groupby([\"make\",\"capacity\",\"failure\"]).size().unstack()\n",
    "gp[\"proportion\"] = gp[1]/ gp.sum(axis=1)\n",
    "gp.reset_index(inplace=True)\n",
    "\n"
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    "Kaplan-Meirer Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The Kaplan-Meier model gives a maxium-likelihood estimate of survival function.mro()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "#Kaplan Meier Model\n",
    "def  estimate_cond_mean(S):\n",
    "    \"\"\"Quick est of conditional mean lifetime\"\"\"\n",
    "    fstar = -S.diff() / (1-S.iloc[-1.0])\n",
    "    Sstar = (S-S.iloc[-1,0]) / (1-S.iloc[-1,0])\n",
    "    llstarr = fstar/ Sstar\n",
    "\n",
    "    llstar[pd.isnull(llstar)] = 0\n",
    "    llstar = llstar[np.isfinite(llstar)]\n",
    "    llstarcs = llstar.cumsum().reset_index()\n",
    "    llstarcs[\"timelinediff\"] = np.append(llstarcs[\"timeline\"].diff().value[1:],0)\n",
    "    llstarcs[\"auc\"] = llstarcs[\"timelinediff\"]*llstarcs[\"KM_estimate\"]\n",
    "    return np.nansum(llstarcs[\"auc\"]).round()\n",
    "\n",
    "#Generic plotting\n",
    "def plot_km(km, axes, suptxt='', subtxt='', i=0, j=0, arws=[], xmax=0, smlfs=10):\n",
    "\n",
    "    ax = km.plot(ax=axes[i,j], title=subtxt, legend=False)\n",
    "    plt.suptitle(suptxt, fontsize=14)\n",
    "    axes[i,j].axhline(0.5, ls='--', lw=0.5)\n",
    "    axes[i,j].annotate('half-life', fontsize=smlfs, color='b'\n",
    "            ,xy=(0,0.5), xycoords='axes fraction'\n",
    "            ,xytext=(10,4), textcoords='offset points')\n",
    "\n",
    "    S = km.survival_function_\n",
    "    hl = S.loc[S['KM_estimate']<0.5,'KM_estimate'].head(1)\n",
    "    if len(hl) == 1:\n",
    "        axes[i,j].annotate('{:.0f}'.format(hl.index[0]), fontsize=smlfs\n",
    "            ,xy=(0,0.5), xycoords='axes fraction'\n",
    "            ,xytext=(10,-12), textcoords='offset points', color='b')\n",
    "\n",
    "    for pt in arws:\n",
    "        tml = km.survival_function_[:pt].tail(1)\n",
    "        plt.annotate('{:.1%}\\n@ {:.0f}hrs'.format(tml['KM_estimate'].values[0],tml.index.values[0])\n",
    "                ,xy=(tml.index.values,tml['KM_estimate'].values), xycoords='data'\n",
    "                ,xytext=(6,-50), textcoords='offset points', color='#007777', fontsize=smlfs\n",
    "                ,arrowprops={'facecolor':'#007777', 'width':2})\n",
    "\n",
    "    ax.set_ylim([0,1])\n",
    "    ax.set_xlim([0,xmax])\n",
    "\n",
    "\n",
    "#All hard drives Failure rate\n",
    "fig, axes = plt.subplots(nrows=1, ncols=1\n",
    "                         ,squeeze=False, sharex=True, sharey=True)\n",
    "km = sa.KaplanMeierFitter()\n",
    "km.fit(durations=df['maxhours'], event_observed=df['failure'])\n",
    "plot_km(km, axes, xmax=df.shape[0], arws=[8760, 43830]\n",
    "        ,suptxt='Kaplan Meier fit for all hardrives', smlfs=12)\n",
    "km.plot\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Clearly the 2TB & 3TB harddrive from Seagate are problematic.  "
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    "Life"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "estimate_cond_mean(km.survival_function_)\n",
    "\n",
    "#Lifetime by Make\n",
    "fig,axes = plt.subplots(nrows=1, ncols=len(df[\"make\"].value_counts()),\n",
    "                         squeeze=False, sharex=True, sharey=True)\n",
    "for j, mfr in enumerate(np.unique(df[\"make\"])):\n",
    "    dfsub = df.loc[df[\"make\"]==mfr]\n",
    "    km = sa.KaplanMeierFitter()\n",
    "    km.fit(durations=dfsub[\"maxhours\"], event_observed=dfsub[\"failure\"])\n",
    "    plot_km(km, axes, j=j, subtxt=mfr, xmax=df.shape[0])\n",
    "\n",
    "fig, axes = plt.subplots(nrows=len(df[\"make\"].value_counts()),\n",
    "                         ncols=len(df[\"capacity\"].value_counts()),\n",
    "                         squeeze=False, sharex=True, sharey=True, figsize=(10,10))\n",
    "\n",
    "km.plot\n",
    "plot.show()\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    "Now "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#Lifetime by Make & Capacity\n",
    "fig, axes = plt.subplots(nrows=len(df['make'].value_counts())\n",
    "                         ,ncols=len(df['capacity'].value_counts())\n",
    "                         ,squeeze=False, sharex=True, sharey=True, figsize=(10,10))\n",
    "\n",
    "\n",
    "for i, mfr in enumerate(np.unique(df['make'])):\n",
    "    for j, cap in enumerate(np.unique(df['capacity'])):\n",
    "        dfsub = df.loc[(df['make']==mfr) & (df['capacity']==cap)]\n",
    "        if dfsub.shape[0]!=0:\n",
    "            km = sa.KaplanMeierFitter()\n",
    "            km.fit(durations=dfsub['maxhours'], event_observed=dfsub['failure'])\n",
    "            plot_km(km, axes, i=i, j=j, subtxt='{} {}'.format(mfr, cap), xmax=df.shape[0])\n",
    "            axes[i,j].annotate('Tot: {}'.format(dfsub.shape[0]), xy=(0.5,0.1), xycoords='axes fraction')\n",
    "        else:\n",
    "            axes[i,j].axis('off')\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Clearly the 2TB & 3TB harddrive from Seagate are problematic.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#Just looking at 3TB harddrives by make\n",
    "dd = OrderedDict()\n",
    "fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, sharex=True, sharey=True)\n",
    "\n",
    "for j, mfr in enumerate(np.unique(df[\"make\"])):\n",
    "    dd[mfr]= {8760:\"\",17520:\"\",26280:\"\"}\n",
    "    dfsub = df.loc[(df[\"make\"]==mfr) & (df[\"capacity\"]==3.0)]\n",
    "    km = sa.KaplanMeierFitter()\n",
    "    km.fit(durations=dfsub[\"maxhours\"], event_observed=dfsub[\"failure\"])\n",
    "    ax = km.plot(ax=axes[0,0], legend=False)\n",
    "    axes[0,0].axhline(0.5, ls='--', lw=0.5)\n",
    "    axes[0,0].annotate('half-life', xy=(0,0.5), xycoords='axes fraction'\n",
    "                ,xytext=(10,4), textcoords='offset points', color='b', fontsize=10)\n",
    "    ax.set_ylim([0,1])\n",
    "    ax.set_xlim([0,df.shape[0]])\n",
    "    fnlS = km.survival_function_.iloc[-1:, :]\n",
    "    axes[0,0].annotate('{}'.format(mfr), xy=(fnlS.index.values[0],fnlS.values[0][0])\n",
    "                       ,xycoords='data', fontsize=12\n",
    "                       ,xytext=(10,0), textcoords='offset points')\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that in nearly 22k hours (2.6yrs), 50% of Seagate harddrives we would expect to fail.  Compare this with 98% of Hitachi drives still working after 2.6yrs.  "
   ]
  },
  {
   "cell_type": "heading",
   "metadata": {},
   "level": 1,
   "source": [
    "Cox Hazard Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The Cox Proportional Hazard model predicts the survival or hazard rate at each point in time.  This gives a semi-parametric method for estimating the hazard function against the covariates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#Cox Hazard Model\n",
    "modelspec = \"make + capacity\"\n",
    "dft = pt.dmatrix(modelspec, df, return_type=\"dataframe\")\n",
    "design_info = dft.design_info\n",
    "dft = dft.join(df[[\"maxhours\",\"failure\"]])\n",
    "\n",
    "del dft[\"Intercept\"]\n",
    "cx = sa.CoxPHFitter()\n",
    "cx.fit(df=dft, duration_col='maxhours', event_col='failure'\n",
    "           ,show_progress=True, include_likelihood=True)\n",
    "\n",
    "fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, sharex=True)\n",
    "cx.baseline_cumulative_hazard_.plot(ax=axes[0,0], legend=False\n",
    "                ,title='Baseline cumulative hazard rate')\n",
    "cx.baseline_survival_.plot(ax=axes[0,1], legend=False\n",
    "                ,title='Baseline survival rate')\n",
    "\n",
    "cx.summary\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    ""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    ""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    ""
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2.0
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}