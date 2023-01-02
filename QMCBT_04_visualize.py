# BUILD A FUNCTION THAT DOES THIS FOR ALL "FLOAT" COLUMNS

# float_cols = train_iris.select_dtypes(include='float').columns

# Plot numeric columns
#plot_float_cols = float_cols 
#for col in plot_float_cols:
#    plt.hist(train_iris[col])
#    plt.title(col)
#    plt.show()
#    plt.boxplot(train_iris[col])
#    plt.title(col)
#    plt.show()


# In[ ]:


# BUILD A FUNCTION THAT DOES THIS FOR ALL "OBJECT" COLUMNS

# train.species.value_counts()
# plt.hist(train_iris.species_name)


# In[ ]:


# BUILD A FUNCTION THAT DOES THIS

#test_var = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#for var in test_var:
#    t_stat, p_val = t_stat, p_val = stats.mannwhitneyu(virginica[var], versicolor[var], alternative="two-sided")
#    print(f'Comparing {var} between Virginica and Versicolor')
#    print(t_stat, p_val)
#    print('')
#    print('---------------------------------------------------------------------')
#    print('')


# In[ ]:


# sns.pairplot(DF, hue='TARGET_COLUMN', corner=True)
# plt.show()

