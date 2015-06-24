package edu.zju.dd.codeforce.db;

import java.sql.Connection;
import java.sql.SQLException;

/**
 * thread safe
 * @author zjut_DD
 *
 */
public class VisitedPageSql {
	private static final String dbName = "localhost/codeforce";
	private static final String tableName = "visited_page";

	private static Connection connection = null;
	static {
		try {
			Class.forName("com.mysql.jdbc.Driver");
			connection = java.sql.DriverManager.getConnection("jdbc:mysql://" + dbName
					+ "?useUnicode=true&characterEncoding=utf-8", "root", "lovelygirl");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static java.sql.Statement getStatement() {
		try {
			return connection.createStatement();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static boolean isVisited(int pageId) throws SQLException {
		java.sql.ResultSet rs = getStatement().executeQuery("select pageId from "
				+tableName+" where pageId = "+pageId);
		return rs.next();
	}
	
	public static void insert(int pageId) throws SQLException {
		getStatement().execute("insert into " + tableName + " values (" + pageId + ")");
	}
	
	
	public static void main(String[] args) throws Exception{
		for(int i=1;i<=316;i++) {
			insert(i);
		}
		/*
		Scanner scanner = new Scanner(new File("e:/status_log.out"));
		while(scanner.hasNext()) {
			String line = scanner.nextLine();
			tringBuilder sb = new StringBuilder(line);
			for(int i=0;i<sb.length();i++) {
				if(sb.charAt(i) == ',') {
					sb.setCharAt(i, ' ');
				}
			}
			Scanner lScanner = new Scanner(sb.toString());
			lScanner.next();
			lScanner.next();
			insert(lScanner.nextInt());
		}*/
		
		long t = System.currentTimeMillis();
		System.out.println(isVisited(39917));
		System.out.println(System.currentTimeMillis() - t);
	}
}
